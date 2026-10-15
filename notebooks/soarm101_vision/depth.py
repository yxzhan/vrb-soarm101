"""Monocular depth estimation of ROS camera streams.

Subscribes to cameras' ``sensor_msgs/CompressedImage`` topics - the
``/<camera>/image_raw/compressed`` topics bambot's camera panel and the MuJoCo
sim publish - runs a YOLO26 depth model on the newest frame of each, and
publishes per camera:

- ``/<camera>_depth/image_raw/compressed``: the depth map coloured (JPEG, near
  is warm, far is cool). Its name follows the camera topic pattern, so bambot's
  camera panel lists it as another device's camera.
- ``/<camera>/depth/compressed``: the depth itself, a 16-bit PNG in millimetres
  (0 = no depth), the same size as the camera image.

The depth is metric, but from a single image: its absolute scale is the
model's guess, which can be off by tens of percent away from the scenes it
was trained on (relative depth within a frame is much better).

Frames that arrive while a frame is being processed are dropped, so the
output never lags behind the camera. With ``--all-cameras`` every camera on
the graph is used (except the vision nodes' own ``*_seg`` and ``*_depth``
outputs), found again every second.

Models: ``yolo26{n,s,m,l,x}-depth.pt`` (needs ultralytics >= 8.4.177); weights
download to ``~/.cache/soarm101_vision``.

Run (the ROS environment sourced)::

    python3 notebooks/soarm101_vision/depth.py --camera camera_1
    python3 notebooks/soarm101_vision/depth.py --all-cameras --model yolo26m-depth.pt
"""

import argparse
import contextlib
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

from camera_topics import discover_cameras

WEIGHTS_DIR = Path.home() / ".cache" / "soarm101_vision"
JPEG_QUALITY = 85
COLOR_SUFFIX = "_depth"
# Cameras are looked for this often; the first look, right at start, usually
# finds nothing yet - the ROS graph takes a moment to discover.
DISCOVERY_PERIOD_S = 1.0
# Largest depth the 16-bit millimetre PNG holds, metres.
MAX_DEPTH_M = 65.535


def load_model(name: str):
    """Loads an Ultralytics depth model, downloading it if needed."""
    from ultralytics import YOLO

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(name).expanduser()
    if not path.is_absolute() and not path.exists():
        path = WEIGHTS_DIR / name
    with contextlib.chdir(WEIGHTS_DIR):
        model = YOLO(str(path))
    if model.task != "depth":
        raise ValueError(f"{name} is a {model.task} model, not a depth model")
    return model


def colorize(depth: np.ndarray, near: Optional[float], far: Optional[float]) -> np.ndarray:
    """Depth (metres) as a BGR image: turbo colour map, near warm and far cool.

    Args:
        depth: (h, w) depth, metres.
        near, far: Range the colours span, metres; by default the frame's
            2nd and 98th percentiles.
    """
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros((*depth.shape, 3), np.uint8)
    lo = near if near is not None else float(np.percentile(valid, 2))
    hi = far if far is not None else float(np.percentile(valid, 98))
    scaled = np.clip((hi - depth) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


class CameraStream:
    """One camera's input subscription, newest unprocessed frame and outputs."""

    def __init__(self, node: Node, camera: str, input_topic: str) -> None:
        self.camera = camera
        self.input_topic = input_topic
        self.latest: Optional[CompressedImage] = None
        self.subscription = node.create_subscription(
            CompressedImage, input_topic, self._on_image, qos_profile_sensor_data
        )
        self.color_pub = node.create_publisher(
            CompressedImage, f"/{camera}{COLOR_SUFFIX}/image_raw/compressed", 1
        )
        self.depth_pub = node.create_publisher(CompressedImage, f"/{camera}/depth/compressed", 1)

    def _on_image(self, msg: CompressedImage) -> None:
        # Only the newest frame matters; an older unprocessed one is dropped.
        self.latest = msg


class DepthNode(Node):
    """Estimates the depth of the newest frame of camera topics and publishes it.

    Args:
        cameras: Camera names; topics are derived from them.
        model: A loaded Ultralytics depth model.
        input_topic: Overrides ``/<camera>/image_raw/compressed`` (one camera only).
        all_cameras: Also use every camera topic found on the graph.
        image_size: Inference size, px (the long side).
        device: Torch device, e.g. ``"0"`` or ``"cpu"``.
        max_hz: Most frames processed per second, per camera.
        near, far: Range of the colour map, metres (None: per frame).
    """

    def __init__(
        self,
        cameras: List[str],
        model,
        input_topic: Optional[str],
        all_cameras: bool,
        image_size: int,
        device: str,
        max_hz: float,
        near: Optional[float],
        far: Optional[float],
    ) -> None:
        super().__init__("depth")
        self._model = model
        self._image_size = image_size
        self._device = device
        self._near = near
        self._far = far
        self._streams: Dict[str, CameraStream] = {}
        self._frames = 0
        self._busy_s = 0.0

        self.get_logger().info(
            f"Model {Path(model.ckpt_path or 'model').name} on device {device}"
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
        self.get_logger().info(f"Estimating the depth of {input_topic}")

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
                frame, imgsz=self._image_size, device=self._device, verbose=False
            )[0]
            depth = result.depth.data.cpu().numpy().astype(np.float32)
            if depth.shape != frame.shape[:2]:
                depth = cv2.resize(
                    depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            self._publish(stream, msg, depth)
            self._busy_s += time.perf_counter() - started
            self._frames += 1

    def _publish(self, stream: CameraStream, msg: CompressedImage, depth: np.ndarray) -> None:
        colored = colorize(depth, self._near, self._far)
        ok, jpeg = cv2.imencode(".jpg", colored, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            stream.color_pub.publish(self._compressed(msg, "jpeg", jpeg))

        millimetres = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        millimetres = np.round(np.clip(millimetres, 0.0, MAX_DEPTH_M) * 1000.0).astype(np.uint16)
        ok, png = cv2.imencode(".png", millimetres)
        if ok:
            stream.depth_pub.publish(self._compressed(msg, "16UC1; png, mm", png))

    @staticmethod
    def _compressed(source: CompressedImage, kind: str, data: np.ndarray) -> CompressedImage:
        out = CompressedImage()
        out.header = source.header
        out.format = kind
        out.data = data.tobytes()
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
    parser = argparse.ArgumentParser(description="Monocular depth of ROS camera streams")
    parser.add_argument(
        "--camera",
        action="append",
        help="Camera name: reads /<camera>/image_raw/compressed; repeat for more cameras "
        "(default: camera_1, or none with --all-cameras)",
    )
    parser.add_argument(
        "--all-cameras",
        action="store_true",
        help="Use every /<name>/image_raw/compressed camera, including ones that start later",
    )
    parser.add_argument("--topic", help="Input topic, overriding the one --camera gives")
    parser.add_argument(
        "--model",
        default="yolo26s-depth.pt",
        help="Ultralytics depth weights (default: yolo26s-depth.pt)",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference size, px (default: 640)")
    parser.add_argument("--device", help="Torch device (default: GPU 0 if available, else cpu)")
    parser.add_argument(
        "--max-hz",
        type=float,
        default=20.0,
        help="Most frames processed per second, per camera (default: 10)",
    )
    parser.add_argument(
        "--near", type=float, help="Depth the colour map starts at, metres (default: per frame)"
    )
    parser.add_argument(
        "--far", type=float, help="Depth the colour map ends at, metres (default: per frame)"
    )
    # ROS arguments (--ros-args ...) go to rclpy; anything else must be ours.
    args = parser.parse_args(remove_ros_args(sys.argv)[1:])
    cameras = args.camera or ([] if args.all_cameras else ["camera_1"])
    if args.topic and (len(cameras) != 1 or args.all_cameras):
        parser.error("--topic needs exactly one --camera and no --all-cameras")

    # Loading can take a while (a download on first use; YOLOE's text encoder)
    # before ROS logs anything.
    print(f"Loading {args.model}...", flush=True)
    model = load_model(args.model)

    rclpy.init(args=sys.argv)
    node = DepthNode(
        cameras=cameras,
        model=model,
        input_topic=args.topic,
        all_cameras=args.all_cameras,
        image_size=args.imgsz,
        device=args.device or default_device(),
        max_hz=args.max_hz,
        near=args.near,
        far=args.far,
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
