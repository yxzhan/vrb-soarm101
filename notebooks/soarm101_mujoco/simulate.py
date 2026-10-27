"""SOARM101 Mujoco Simulation.

Uses MuJoCo Menagerie's SO101 model on a table (SO101/scene.xml).
With ROS, the scene's cameras (``front`` and ``right``, 640x480) are published as
JPEG on ``/sim_<camera>/image_raw/compressed``, the topic pattern bambot's camera
panel and ``soarm101_vision/segment.py`` read.
"""

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Camera images render offscreen through EGL (GPU, no window); the viewer keeps
# its own GLFW window. Must be set before mujoco is imported.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import mujoco.viewer


def get_package_dir() -> Path:
    return Path(__file__).parent


JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

HOME_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
    "gripper": 0.785,
}

# How often the simulated joint state is published, seconds (~30 Hz).
STATE_PUBLISH_PERIOD = 1 / 30
# How often the camera images are published, seconds (~15 Hz).
CAMERA_PUBLISH_PERIOD = 1 / 15
# How often the viewer is synced, seconds (~60 Hz). Physics steps in between to keep
# up with the wall clock: syncing every step would tie the sim's speed to the
# viewer's rendering (software OpenGL over VNC: several ms a frame, a fraction of
# real time).
VIEWER_SYNC_PERIOD = 1 / 60
# Most physics steps per loop when behind (e.g. after a stall): the sim slows down
# rather than spending ever longer catching up.
MAX_STEPS_PER_LOOP = 100
CAMERA_TOPIC = "/sim_{}/image_raw/compressed"
JPEG_QUALITY = 85


class CameraPublisher:
    """Renders the model's cameras offscreen and publishes them as JPEG.

    Args:
        model: The model; every camera in it given a ``resolution`` is
            published, at that resolution (the arm's ``wrist_cam`` has none).
        ros_bridge: The running ROS bridge.
    """

    def __init__(self, model: mujoco.MjModel, ros_bridge) -> None:
        self._ros_bridge = ros_bridge
        self._renderers = {}
        for i in range(model.ncam):
            width, height = (int(v) for v in model.cam_resolution[i])
            if width > 1 and height > 1:
                self._renderers[model.camera(i).name] = mujoco.Renderer(model, height, width)
        self._cameras = list(self._renderers)

    @property
    def topics(self) -> list:
        return [CAMERA_TOPIC.format(name) for name in self._cameras]

    def publish(self, data: mujoco.MjData) -> None:
        for name, renderer in self._renderers.items():
            renderer.update_scene(data, camera=name)
            rgb = renderer.render()
            ok, jpeg = cv2.imencode(
                ".jpg", rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                self._ros_bridge.publish_image(CAMERA_TOPIC.format(name), name, jpeg.tobytes())

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()


def load_model() -> mujoco.MjModel:
    xml_path = get_package_dir() / "SO101" / "scene.xml"

    if not xml_path.exists():
        print(f"Error: XML file not found: {xml_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    print(
        f"Loaded: nq={model.nq}, nbody={model.nbody}, ngeom={model.ngeom}",
        flush=True,
    )
    return model


def get_joint_ids(model: mujoco.MjModel) -> dict:
    ids = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            ids[name] = jid
    return ids


def set_joint_positions(data: mujoco.MjData, joint_pos: dict) -> None:
    for name, pos in joint_pos.items():
        jid = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            qpos_adr = data.model.jnt_qposadr[jid]
            data.qpos[qpos_adr] = pos


def run_simulation(
    model: mujoco.MjModel,
    use_ros: bool = False,
    command_topic: str = "/joint_command",
    state_topic: Optional[str] = "/joint_states",
    cameras: bool = True,
) -> None:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    for name, pos in HOME_POSITION.items():
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            data.ctrl[aid] = pos
    set_joint_positions(data, HOME_POSITION)
    mujoco.mj_forward(model, data)

    joint_ids = get_joint_ids(model)
    print(f"Robot joint IDs: {joint_ids}", flush=True)


    ros_bridge = None
    if use_ros:
        try:
            spec = importlib.util.spec_from_file_location(
                "ros_bridge", get_package_dir() / "ros_bridge.py"
            )
            ros_bridge_mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ros_bridge_mod)
            RosBridge = ros_bridge_mod.RosBridge
            ros_bridge = RosBridge(command_topic, state_topic)
            ros_bridge.start()
            print(
                f"[Simulation] ROS 2 bridge enabled - following {command_topic}"
                + (f", publishing {state_topic}" if state_topic else ""),
                flush=True,
            )
        except Exception as e:
            print(
                f"[Simulation] Failed to start ROS bridge: {e}",
                flush=True,
            )
            print(
                "[Simulation] Running without ROS control", flush=True
            )
            use_ros = False

    camera_publisher = None
    if ros_bridge is not None and cameras and model.ncam:
        try:
            camera_publisher = CameraPublisher(model, ros_bridge)
            print(
                f"[Simulation] Publishing cameras on {', '.join(camera_publisher.topics)}",
                flush=True,
            )
        except Exception as e:
            print(f"[Simulation] Cameras off, offscreen rendering failed: {e}", flush=True)

    def set_actuator_targets(data: mujoco.MjData, targets: dict) -> None:
        for name, pos in targets.items():
            aid = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = pos

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        last_state_publish = 0.0
        last_camera_publish = 0.0
        # Wall clock time at which the sim's time was 0.
        wall_start = time.monotonic() - data.time
        while viewer.is_running():
            if use_ros and ros_bridge is not None:
                ros_positions = ros_bridge.get_joint_positions()
                if ros_positions:
                    set_actuator_targets(data, ros_positions)

            # Step until the sim's time catches up with the wall clock.
            target = time.monotonic() - wall_start + VIEWER_SYNC_PERIOD
            steps = 0
            while data.time < target and steps < MAX_STEPS_PER_LOOP:
                mujoco.mj_step(model, data)
                steps += 1
            if steps == MAX_STEPS_PER_LOOP:
                wall_start = time.monotonic() - data.time  # fell behind: slow down
            viewer.sync()
            ahead = data.time - (time.monotonic() - wall_start)
            if ahead > 0:
                time.sleep(ahead)

            now = time.monotonic()
            if ros_bridge is not None and now - last_state_publish >= STATE_PUBLISH_PERIOD:
                last_state_publish = now
                names = list(joint_ids)
                positions = [
                    float(data.qpos[model.jnt_qposadr[joint_ids[name]]]) for name in names
                ]
                ros_bridge.publish_state(names, positions)
            if camera_publisher is not None and now - last_camera_publish >= CAMERA_PUBLISH_PERIOD:
                last_camera_publish = now
                camera_publisher.publish(data)

    if camera_publisher is not None:
        camera_publisher.close()
    if ros_bridge is not None:
        ros_bridge.stop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SOARM101 Mujoco Simulation"
    )
    parser.add_argument(
        "--ros",
        action="store_true",
        default=True,
        help="Enable the ROS 2 bridge (default: enabled)",
    )
    parser.add_argument(
        "--command-topic",
        default="/joint_command",
        help="Topic with target joint positions to follow (default: /joint_command)",
    )
    parser.add_argument(
        "--state-topic",
        default="/joint_states",
        help="Topic to publish the simulated joint state on (default: /joint_states)",
    )
    parser.add_argument(
        "--no-state",
        action="store_true",
        help="Do not publish the simulated state, e.g. when a real follower owns the state topic",
    )
    parser.add_argument(
        "--no-cameras",
        action="store_true",
        help="Do not render and publish the scene's cameras",
    )

    args = parser.parse_args()

    model = load_model()
    run_simulation(
        model,
        use_ros=args.ros,
        command_topic=args.command_topic,
        state_topic=None if args.no_state else args.state_topic,
        cameras=not args.no_cameras,
    )


if __name__ == "__main__":
    main()
