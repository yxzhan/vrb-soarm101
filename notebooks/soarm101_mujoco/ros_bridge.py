"""ROS 2 bridge for SOARM101 Mujoco simulation.

The simulation plays the follower: it takes target joint positions from the
command topic (``/joint_commands``, published by a bambot leader) and
publishes the simulated arm's actual joint positions on the state topic
(``/joint_states``), the same contract as a bambot page in the follower role.
"""

import threading
from typing import Dict, List, Optional

try:
    import rclpy
    import rclpy.executors
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[ROS Bridge] rclpy not available - ROS control disabled")


DEFAULT_COMMAND_TOPIC = "/joint_commands"
DEFAULT_STATE_TOPIC = "/joint_states"


if ROS2_AVAILABLE:
    class JointStateBridge(Node):
        def __init__(self, command_topic: str, state_topic: Optional[str]) -> None:
            super().__init__("soarm101_mujoco_bridge")
            self.joint_positions: Dict[str, float] = {}
            self._lock = threading.Lock()
            self._subscription = self.create_subscription(
                JointState,
                command_topic,
                self._command_callback,
                1,
            )
            self._state_publisher = (
                self.create_publisher(JointState, state_topic, 10)
                if state_topic
                else None
            )
            self.get_logger().info("SOARM101 Mujoco ROS Bridge started")
            self.get_logger().info(f"Following commands on {command_topic}")
            if state_topic:
                self.get_logger().info(f"Publishing simulated state on {state_topic}")

        def _command_callback(self, msg: JointState) -> None:
            with self._lock:
                for name, pos in zip(msg.name, msg.position):
                    self.joint_positions[name] = pos

        def get_joint_positions(self) -> Dict[str, float]:
            with self._lock:
                return dict(self.joint_positions)

        def publish_state(self, names: List[str], positions: List[float]) -> None:
            if self._state_publisher is None:
                return
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.name = names
            msg.position = positions
            self._state_publisher.publish(msg)


class RosBridge:
    """Runs the bridge node on a background thread.

    Args:
        command_topic: Topic carrying target joint positions.
        state_topic: Topic to publish the simulated joint state on, or None to
            stay silent (e.g. when a real follower already owns that topic).
    """

    def __init__(
        self,
        command_topic: str = DEFAULT_COMMAND_TOPIC,
        state_topic: Optional[str] = DEFAULT_STATE_TOPIC,
    ) -> None:
        if not ROS2_AVAILABLE:
            raise RuntimeError("ROS 2 is not available")
        self._command_topic = command_topic
        self._state_topic = state_topic
        self._node: Optional[JointStateBridge] = None
        self._executor: Optional[rclpy.executors.SingleThreadedExecutor] = None
        self._spin_thread: Optional[threading.Thread] = None
        self._running = False

    def start(self) -> None:
        rclpy.init()
        self._node = JointStateBridge(self._command_topic, self._state_topic)
        self._executor = rclpy.executors.SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._running = True
        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()

    def _spin(self) -> None:
        while self._running and rclpy.ok():  # type: ignore[attr-defined]
            if self._executor is not None:
                self._executor.spin_once(timeout_sec=0.01)

    def get_joint_positions(self) -> Dict[str, float]:
        if self._node is None:
            return {}
        return self._node.get_joint_positions()

    def publish_state(self, names: List[str], positions: List[float]) -> None:
        if self._node is not None:
            self._node.publish_state(names, positions)

    def stop(self) -> None:
        self._running = False
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        if self._node is not None and self._executor is not None:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        rclpy.shutdown()  # type: ignore[attr-defined]
