"""Instance segmentation of a ROS camera stream.

Subscribes to a camera's ``sensor_msgs/CompressedImage`` topic - the
``/<camera>/image_raw/compressed`` topics bambot's camera panel publishes - runs
an Ultralytics segmentation model on the newest frame, and publishes:

- ``/<camera>_seg/image_raw/compressed``: the frame with the masks drawn on it
  (JPEG). Its name follows the camera topic pattern, so bambot's camera panel
  lists it as another device's camera.
- ``/<camera>/segmentation``: ``std_msgs/String`` with JSON - per object its
  class, score, bounding box, centroid, area and outline, in image pixels.
- ``/<camera>/segmentation/mask/compressed``: a PNG label image, 0 for
  background and ``i + 1`` for the i-th object in the JSON.

Frames that arrive while a frame is being processed are dropped, so the
output never lags behind the camera.

With ``--all-cameras`` it segments every camera on the graph: each
``/<name>/image_raw/compressed`` topic (except the vision nodes' own ``*_seg``
and ``*_depth`` outputs), found
again every second so cameras that come later are picked up too. The
cameras share one model.

Models: any Ultralytics ``*-seg`` model. The default, YOLO11n-seg, knows the 80
COCO classes. YOLOE (e.g. ``yoloe-11s-seg.pt``) takes the classes as text with
``--classes``, for objects COCO does not cover; its text encoder is a 570 MB
download on first use. Weights download to ``~/.cache/soarm101_vision``.

Run (the ROS environment sourced)::

    python3 notebooks/soarm101_vision/segment.py --camera camera_1
    python3 notebooks/soarm101_vision/segment.py --all-cameras
    python3 notebooks/soarm101_vision/segment.py --camera camera_1 \\
        --model yoloe-11s-seg.pt --classes "cup, red block, tape"
"""

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.utilities import remove_ros_args
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String

from camera_topics import discover_cameras

WEIGHTS_DIR = Path.home() / ".cache" / "soarm101_vision"
JPEG_QUALITY = 85
# Outline simplification tolerance, px: keeps the JSON small without
# visibly changing the shape.
POLYGON_EPSILON_PX = 1.5
# Suffix of the overlay camera this node publishes (see camera_topics).
OVERLAY_SUFFIX = "_seg"
# Cameras are looked for this often; the first look, right at start, usually
# finds nothing yet - the ROS graph takes a moment to discover.
DISCOVERY_PERIOD_S = 1.0


def load_model(name: str, classes: List[str]):
    """Loads an Ultralytics segmentation model, downloading it if needed.

    Args:
        name: Weights file name (fetched from Ultralytics' releases) or path.
        classes: Text prompts for a YOLOE model; must be empty for other models.
    """
    from ultralytics import YOLO, YOLOE

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(name).expanduser()
    if not path.is_absolute() and not path.exists():
        path = WEIGHTS_DIR / name
    is_yoloe = "yoloe" in path.name.lower()
    if classes and not is_yoloe:
        raise ValueError("--classes needs a YOLOE model, e.g. yoloe-11s-seg.pt")
    # Ultralytics downloads helper files (YOLOE's text encoder) into the
    # working directory; keep them with the weights.
    with contextlib.chdir(WEIGHTS_DIR):
        model = (YOLOE if is_yoloe else YOLO)(str(path))
        if classes:
            model.set_classes(classes, model.get_text_pe(classes))
    if model.task != "segment":
        raise ValueError(f"{name} is a {model.task} model, not a segmentation model")
    return model


class CameraStream:
    """One camera's input subscription, newest unprocessed frame and outputs."""

    def __init__(self, node: Node, camera: str, input_topic: str) -> None:
        self.camera = camera
        self.input_topic = input_topic
        self.latest: Optional[CompressedImage] = None
        self.subscription = node.create_subscription(
            CompressedImage, input_topic, self._on_image, qos_profile_sensor_data
        )
        self.overlay_pub = node.create_publisher(
            CompressedImage, f"/{camera}{OVERLAY_SUFFIX}/image_raw/compressed", 1
        )
        self.objects_pub = node.create_publisher(String, f"/{camera}/segmentation", 10)
        self.mask_pub = node.create_publisher(
            CompressedImage, f"/{camera}/segmentation/mask/compressed", 1
        )

    def _on_image(self, msg: CompressedImage) -> None:
        # Only the newest frame matters; an older unprocessed one is dropped.
        self.latest = msg


class SegmentationNode(Node):
    """Segments the newest frame of camera topics and publishes the results.

    Args:
        cameras: Camera names; topics are derived from them.
        model: A loaded Ultralytics segmentation model.
        input_topic: Overrides ``/<camera>/image_raw/compressed`` (one camera only).
        all_cameras: Also segment every camera topic found on the graph.
        confidence: Detections below this score are dropped.
        image_size: Inference size, px (the long side).
        device: Torch device, e.g. ``"0"`` or ``"cpu"``.
        max_hz: Most frames processed per second, per camera.
    """

    def __init__(
        self,
        cameras: List[str],
        model,
        input_topic: Optional[str],
        all_cameras: bool,
        confidence: float,
        image_size: int,
        device: str,
        max_hz: float,
    ) -> None:
        super().__init__("segmentation")
        self._model = model
        self._confidence = confidence
        self._image_size = image_size
        self._device = device
        self._streams: Dict[str, CameraStream] = {}
        self._frames = 0
        self._busy_s = 0.0

        self.get_logger().info(
            f"Model {Path(model.ckpt_path or 'model').name} "
            f"({len(model.names)} classes) on device {device}"
        )
        for camera in cameras:
            self._add_camera(camera, input_topic or f"/{camera}/image_raw/compressed")
        if all_cameras:
            self._discover()
            self.create_timer(DISCOVERY_PERIOD_S, self._discover)
        self.create_timer(1.0 / max_hz, self._process_latest)
        self.create_timer(10.0, self._report)

    def _add_camera(self, camera: str, input_topic: str) -> None:
        if camera in self._streams:
            return
        self._streams[camera] = CameraStream(self, camera, input_topic)
        self.get_logger().info(f"Segmenting {input_topic}")

    def _discover(self) -> None:
        for camera in discover_cameras(self):
            self._add_camera(camera, f"/{camera}/image_raw/compressed")

    def _process_latest(self) -> None:
        for stream in self._streams.values():
            msg, stream.latest = stream.latest, None
            if msg is None:
                continue
            frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                self.get_logger().warning(
                    f"Could not decode a '{msg.format}' frame from {stream.input_topic}"
                )
                continue
            started = time.perf_counter()
            result = self._model.predict(
                frame,
                conf=self._confidence,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )[0]
            self._publish(stream, msg, frame, result)
            self._busy_s += time.perf_counter() - started
            self._frames += 1

    def _publish(
        self, stream: CameraStream, msg: CompressedImage, frame: np.ndarray, result
    ) -> None:
        height, width = frame.shape[:2]
        objects = []
        outlines = []
        if result.masks is not None:
            boxes = result.boxes
            for i, outline in enumerate(result.masks.xy):
                if len(outline) < 3 or i >= 255:
                    continue
                polygon = outline.astype(np.float32).reshape(-1, 1, 2)
                moments = cv2.moments(polygon)
                if moments["m00"] <= 0:
                    continue
                outlines.append(np.round(polygon).astype(np.int32))
                simplified = cv2.approxPolyDP(polygon, POLYGON_EPSILON_PX, True).reshape(-1, 2)
                class_id = int(boxes.cls[i])
                objects.append(
                    {
                        "id": len(objects) + 1,
                        "class": result.names[class_id],
                        "class_id": class_id,
                        "score": round(float(boxes.conf[i]), 3),
                        "bbox": [round(float(v), 1) for v in boxes.xyxy[i]],
                        "centroid": [
                            round(moments["m10"] / moments["m00"], 1),
                            round(moments["m01"] / moments["m00"], 1),
                        ],
                        "area": round(float(moments["m00"]), 1),
                        "polygon": simplified.round(1).tolist(),
                    }
                )

        # Where masks overlap, the smaller object is usually the one in front
        # (a cup on a box): paint the largest first so the small ones stay.
        labels = np.zeros((height, width), np.uint8)
        for obj in sorted(objects, key=lambda o: -o["area"]):
            cv2.fillPoly(labels, [outlines[obj["id"] - 1]], obj["id"])

        report = {
            "stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
            "frame_id": msg.header.frame_id,
            "width": width,
            "height": height,
            "objects": objects,
        }
        stream.objects_pub.publish(String(data=json.dumps(report)))

        overlay = result.plot(img=frame.copy(), conf=True, boxes=True, masks=True)
        stream.overlay_pub.publish(self._compressed(msg, "jpeg", overlay))
        stream.mask_pub.publish(self._compressed(msg, "png", labels))

    @staticmethod
    def _compressed(source: CompressedImage, kind: str, image: np.ndarray) -> CompressedImage:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if kind == "jpeg" else []
        ok, data = cv2.imencode(f".{'jpg' if kind == 'jpeg' else 'png'}", image, params)
        out = CompressedImage()
        out.header = source.header
        out.format = kind
        out.data = data.tobytes() if ok else b""
        return out

    def _report(self) -> None:
        if self._frames:
            self.get_logger().info(
                f"{self._frames / 10.0:.1f} frames/s over {len(self._streams)} camera(s), "
                f"{1000.0 * self._busy_s / self._frames:.0f} ms each"
            )
        self._frames, self._busy_s = 0, 0.0


def default_device() -> str:
    import torch

    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Instance segmentation of a ROS camera stream")
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera name: reads /<camera>/image_raw/compressed; repeat for more cameras "
        "(default: camera_1, or none with --all-cameras)",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Segment every /<name>/image_raw/compressed camera, including ones that start later",
    )
    parser.add_argument("--topic", help="Input topic, overriding the one --camera gives")
    parser.add_argument(
        "--model",
        default="yolo26s-seg.pt",
        help="Ultralytics segmentation weights (default: yolo26s-seg.pt, COCO classes)",
    )
    parser.add_argument(
        "--classes",
        default="",
        help='Comma separated text prompts for a YOLOE model, e.g. "cup, red block"',
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum score (default: 0.25)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size, px (default: 640)")
    parser.add_argument("--device", help="Torch device (default: GPU 0 if available, else cpu)")
    parser.add_argument(
        "--max-hz",
        type=float,
        default=20.0,
        help="Most frames processed per second, per camera (default: 10)",
    )
    # ROS arguments (--ros-args ...) go to rclpy; anything else must be ours.
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    cameras = args.camera or ([] if args.all_cameras else ["camera_1"])
    if args.topic and (len(cameras) != 1 or args.all_cameras):
        parser.error("--topic needs exactly one --camera and no --all-cameras")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    # Loading can take a while (a download on first use; YOLOE's text encoder)
    # before ROS logs anything.
    print(f"Loading {args.model}...", flush=True)
    model = load_model(args.model, classes)

    rclpy.init(args=sys.argv)
    node = SegmentationNode(
        cameras=cameras,
        model=model,
        input_topic=args.topic,
        all_cameras=args.all_cameras,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device or default_device(),
        max_hz=args.max_hz,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
