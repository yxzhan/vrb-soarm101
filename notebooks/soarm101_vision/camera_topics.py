"""Finding the camera topics on the ROS graph, shared by the vision nodes."""

import re
from typing import List

from rclpy.node import Node

# Camera topics: /<name>/image_raw/compressed.
CAMERA_TOPIC = re.compile(r"^/([^/]+)/image_raw/compressed$")
# Name suffixes of the cameras the vision nodes publish themselves (the
# segmentation overlay, the coloured depth): never taken as input.
OUTPUT_SUFFIXES = ("_seg", "_depth")


def discover_cameras(node: Node) -> List[str]:
    """Names of the cameras with a publisher, other than the vision nodes' own outputs."""
    cameras = []
    for topic, types in node.get_topic_names_and_types():
        match = CAMERA_TOPIC.match(topic)
        if (
            match
            and "sensor_msgs/msg/CompressedImage" in types
            and not match.group(1).endswith(OUTPUT_SUFFIXES)
            # Only topics someone publishes, not ones merely subscribed to.
            and node.count_publishers(topic) > 0
        ):
            cameras.append(match.group(1))
    return cameras
