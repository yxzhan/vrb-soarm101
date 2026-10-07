"""SOARM101 Mujoco Simulation.

Uses official SOARM101 Mujoco model (SO101/scene.xml).
"""

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Optional

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
    command_topic: str = "/joint_commands",
    state_topic: Optional[str] = "/joint_states",
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

    def set_actuator_targets(data: mujoco.MjData, targets: dict) -> None:
        for name, pos in targets.items():
            aid = mujoco.mj_name2id(data.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                data.ctrl[aid] = pos

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0

        last_state_publish = 0.0
        while viewer.is_running():
            if use_ros and ros_bridge is not None:
                ros_positions = ros_bridge.get_joint_positions()
                if ros_positions:
                    set_actuator_targets(data, ros_positions)

            mujoco.mj_step(model, data)
            viewer.sync()

            now = time.monotonic()
            if ros_bridge is not None and now - last_state_publish >= STATE_PUBLISH_PERIOD:
                last_state_publish = now
                names = list(joint_ids)
                positions = [
                    float(data.qpos[model.jnt_qposadr[joint_ids[name]]]) for name in names
                ]
                ros_bridge.publish_state(names, positions)

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
        default="/joint_commands",
        help="Topic with target joint positions to follow (default: /joint_commands)",
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

    args = parser.parse_args()

    model = load_model()
    run_simulation(
        model,
        use_ros=args.ros,
        command_topic=args.command_topic,
        state_topic=None if args.no_state else args.state_topic,
    )


if __name__ == "__main__":
    main()
