"""Find which RoboCasa objects the sim arm can pick up.

Each instance of the given categories is put alone on the sim's table in front
of the arm and picked the way a pick skill working from a detected object
would:

1. From the object's collision geometry (what perception would see as points):
   its top, and around the grip height its horizontal slice - the jaws' closing
   direction and the width across it. The grip point (the slice's middle) sits
   ``GRIP_BEHIND_TIP`` back from the fingertips; the fixed jaw's face goes
   ``JAW_STANDOFF`` beside the object, and the moving jaw, opened just wide
   enough, pushes the object against it (both measured from the arm's collision
   geoms, ``measure_jaws``).
2. The tool as steep as the arm allows (``PITCHES``): straight down the jaws
   turn to the slice's narrow side, but the wrist keeps the tool low; tilted,
   it reaches higher, the jaws closing across the direction the arm reaches out.
3. Open, come in along the tool's axis from above the top, close, lift ``LIFT``
   and hold.

Grip heights tried: ``GRIP_BELOW_TOP`` below the top (rims, necks, handles) and
halfway up (bodies), each with the object turned to each of ``YAWS`` (a mug's
handle, a banana's curve). An object is graspable if, at either height, at
least ``MIN_SUCCESSES`` of the turns are lifted.

Objects are converted as ``import_robocasa_objects.py`` does for the scene (same
density, friction and collision geoms), and the arm is the scene's (MuJoCo
Menagerie's SO101: primitive jaw collisions and contact parameters made for
manipulation).

Outcomes: ``ok``; ``knocked`` (the object moved before the jaws closed);
``missed`` (it did not come up, the jaws closed on nothing); ``slipped`` (it
did not come up, or not with the jaws, though they closed on it);
``unreachable`` (no pose for the arm); ``unstable`` (the simulation blew up).

Run (with mujoco; RoboCasa's object assets downloaded)::

    python3 notebooks/soarm101_mujoco/grasp_test.py
    python3 notebooks/soarm101_mujoco/grasp_test.py --categories objaverse/mug --snapshots
"""

import argparse
import csv
import multiprocessing
import os
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from import_robocasa_objects import SO101_DIR, TABLE_Z, TabletopObject, import_object

ARM_XML = SO101_DIR / "so101.xml"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "grasp_results"
# The scene's objects' categories (import_robocasa_objects.OBJECTS) and their
# closest relatives, as <registry>/<category>.
DEFAULT_CATEGORIES = [
    "objaverse/mug",
    "objaverse/cup",
    "objaverse/coffee_cup",
    "lightwheel/glass_cup",
    "lightwheel/measuring_cup",
    "objaverse/bowl",
    "objaverse/plate",
    "objaverse/spoon",
    "objaverse/fork",
    "objaverse/banana",
    "objaverse/apple",
    "objaverse/milk",
    "objaverse/bottled_water",
    "objaverse/water_bottle",
    "objaverse/bottled_drink",
]

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TCP_SITE = "gripperframe"
OBJECT = "object"
# Tool frame (between the jaws): +x points out of the jaws, +z from the fixed jaw
# towards the moving one.
# Lowest the fingertips go above the table, metres.
MIN_TIP_CLEARANCE = 0.002
# Length of the fingers' grasping part, from the tips, metres.
FINGER_DEPTH = 0.035
# The grip point (the object's centre between the jaws) is this far back from the
# fingertips along the fingers, metres.
GRIP_BEHIND_TIP = 0.015
# Heights tried for the grip point: this far below the object's top (rims, necks,
# handles), and halfway up (bodies), metres.
GRIP_BELOW_TOP = 0.02
# The object's width is measured over this far above and below the grip point, metres.
GRIP_SLAB = 0.015
# Gap left between the fixed jaw and the object before closing, metres.
JAW_STANDOFF = 0.005
# Extra clearance for the moving jaw beyond the object when opened, metres.
OPEN_MARGIN = 0.015
# Approach from this far above the object's top, metres.
APPROACH_CLEARANCE = 0.04
# How high the object is lifted, metres; it must come up at least LIFT_PASS of that.
LIFT = 0.08
LIFT_PASS = 0.8
# Most the object may move against the jaws while held, metres.
MAX_SLIP = 0.02
# Most the object may move before the jaws close, metres and degrees.
MAX_KNOCK = 0.015
MAX_KNOCK_TILT = 15.0
# Distances in front of the arm tried for the object, metres (the first reachable wins).
PLACE_X = (0.20, 0.16, 0.24, 0.12)
# Tool pitches tried, degrees below the horizontal, steepest first. Straight down
# (90) the jaws can turn to the object's narrow side, but the wrist's range keeps
# the tool below ~8 cm; tilted, it reaches higher (60: ~16 cm, 45: ~22 cm), the
# jaws closing across the direction the arm reaches out in.
PITCHES = (90, 75, 60, 45, 30)
# Longest straight approach along the tool's axis, metres.
MAX_APPROACH = 0.10
# Most any joint may turn between path points 1 cm apart, rad (more: IK flipped).
MAX_JOINT_JUMP = 0.3
TOOL_SPEED = 0.05  # m/s
SETTLE_S = 0.5
CLOSE_S = 0.6
HOLD_S = 1.0
GRIPPER_CLOSED = -0.17453  # the gripper's lower limit: closing aims past contact
SNAPSHOT_SIZE = (320, 240)
# Turns of the object tried at each grip height, degrees; and how many must be lifted.
YAWS = (0, 120, 240)
MIN_SUCCESSES = 2


def model_xml(source: str, objects_root: Path) -> str:
    """The arm on the scene's table with one object (a free body named ``OBJECT``)."""
    root = ET.Element("mujoco", model="grasp_test")
    ET.SubElement(root, "include", file=str(ARM_XML))
    # After the arm's compiler: its meshdir is relative to its own file.
    ET.SubElement(root, "compiler", meshdir=str(SO101_DIR / "assets"))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "light", pos="0.3 0.3 1.5", dir="-0.2 -0.2 -1", directional="true")
    table = ET.SubElement(worldbody, "body", name="table", pos=f"0.3 0 {TABLE_Z}")
    ET.SubElement(
        table, "geom", type="box", size="0.35 0.6 0.015", pos="0 0 -0.015", rgba="0.93 0.93 0.91 1"
    )
    obj = TabletopObject(OBJECT, source, 1.0, 0.0, 0.0)
    worldbody.append(import_object(obj, objects_root, asset, copy=False))
    ET.SubElement(
        worldbody, "camera", name="snapshot", pos="0.45 -0.3 0.3", mode="targetbody", target=OBJECT
    )
    return ET.tostring(root, encoding="unicode")


class Arm:
    """Kinematics of the arm's five joints and its tool frame, on a scratch MjData.

    Five joints cannot take every tool pose, so a pose is asked for with five
    constraints: the position, and either the tool pointing straight down with
    the jaws closing along a given horizontal direction (one redundant), or the
    tool tilted ``pitch`` below the horizontal with the jaws closing
    horizontally - which way the tool then points is the arm's to choose (the
    plane it reaches out in).
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(TCP_SITE).id
        joints = [model.joint(name) for name in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.lower = np.array([j.range[0] for j in joints])
        self.upper = np.array([j.range[1] for j in joints])
        mujoco.mj_kinematics(model, self.data)
        self.pan_xy = self.data.xanchor[model.joint("shoulder_pan").id][:2].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        # Seeds: elbow up, reaching forward, the wrist rolled every quarter turn.
        self.seeds = [
            np.array([0.0, lift, -lift, flex, roll])
            for lift, flex in ((-0.5, 1.2), (0.3, 1.0), (-1.0, 0.8))
            for roll in (0.0, np.pi / 2, -np.pi / 2, 2.5, -2.5)
        ]

    def tcp(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Tool position and axes (columns) at joint angles ``q``."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        return self.data.site_xpos[self.site].copy(), self.data.site_xmat[self.site].reshape(3, 3)

    def _residual(
        self, q: np.ndarray, pos: np.ndarray, pitch: float, closing: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        p, rot = self.tcp(q)
        x, z = rot[:, 0], rot[:, 2]
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site)
        omega = self._jacr[:, self.dadr]
        dx = np.cross(omega.T, x).T
        dz = np.cross(omega.T, z).T
        if pitch >= 90:
            err = [x[0], x[1], z[0] * closing[1] - z[1] * closing[0]]
            jac = [dx[0], dx[1], dz[0] * closing[1] - dz[1] * closing[0]]
        else:
            err = [x[2] + np.sin(np.radians(pitch)), z[2]]
            jac = [dx[2], dz[2]]
        return (
            np.concatenate([p - pos, err]),
            np.vstack([self._jacp[:, self.dadr], np.array(jac)]),
        )

    def solve(
        self,
        pos: np.ndarray,
        pitch: float,
        closing: np.ndarray,
        seeds: List[np.ndarray],
        iterations: int = 100,
    ) -> Optional[np.ndarray]:
        """Joint angles for the tool at ``pos``, tilted ``pitch`` degrees below the
        horizontal (90 = straight down), the jaws closing along horizontal
        ``closing`` (only its line counts when tilted: the side is checked); or None.
        """
        for seed in seeds:
            q = np.clip(seed, self.lower, self.upper)
            for _ in range(iterations):
                err, jac = self._residual(q, pos, pitch, closing)
                if np.abs(err).max() < 5e-4:
                    if self.tcp(q)[1][:2, 2] @ closing > 0.9:
                        return q
                    break
                step = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(len(err)), err)
                q = np.clip(q - np.clip(step, -0.2, 0.2), self.lower, self.upper)
        return None


@dataclass
class Jaws:
    """The jaws in the tool frame, from their collision geoms, metres."""

    # How far the fixed jaw's tip is beyond the tool frame (along x).
    tip: float
    # Where the fixed jaw's face is (z) over the fingers' grasping length.
    face: float
    # (gripper angle, gap between the jaws over the grasping length), opening up.
    clearances: List[Tuple[float, float]]


def measure_jaws(model: mujoco.MjModel) -> Jaws:
    """The arm's jaws (``Jaws``), measured over ``FINGER_DEPTH`` from the fixed jaw's tip."""
    data = mujoco.MjData(model)
    site = model.site(TCP_SITE).id
    gripper = model.joint("gripper")

    def in_tool(body: str) -> np.ndarray:
        mujoco.mj_kinematics(model, data)
        rot = data.site_xmat[site].reshape(3, 3)
        geoms = [
            g
            for g in np.flatnonzero(model.geom_bodyid == model.body(body).id)
            if model.geom_contype[g] or model.geom_conaffinity[g]
        ]
        points = np.vstack([geom_points(model, data, g) for g in geoms])
        return (points - data.site_xpos[site]) @ rot

    data.qpos[gripper.qposadr[0]] = GRIPPER_CLOSED
    fixed = in_tool("gripper")
    tip = float(fixed[:, 0].max())
    face = float(fixed[fixed[:, 0] > tip - FINGER_DEPTH][:, 2].max())
    clearances = []
    for angle in np.arange(GRIPPER_CLOSED, gripper.range[1] + 1e-6, 0.05):
        data.qpos[gripper.qposadr[0]] = angle
        moving = in_tool("moving_jaw_so101_v1")
        near = moving[moving[:, 0] > tip - FINGER_DEPTH - 0.005]
        gap = near[:, 2].min() - face if len(near) else np.inf
        clearances.append((float(angle), float(gap)))
    return Jaws(tip, face, clearances)


def geom_points(model: mujoco.MjModel, data: mujoco.MjData, geom: int) -> np.ndarray:
    """A geom's extreme points in world coordinates: a mesh's vertices, a box's
    corners, a sphere's or capsule's (or cylinder's) points along its axes."""
    kind, size = model.geom_type[geom], model.geom_size[geom]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        local = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
        local = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * size
    else:
        radius = size[0]
        half = 0.0 if kind == mujoco.mjtGeom.mjGEOM_SPHERE else size[1]
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half += radius
        axes = np.vstack([np.eye(3), -np.eye(3)])
        local = axes * [radius, radius, half if half else radius]
    return data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T


def object_vertices(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """The object's collision vertices in world coordinates."""
    body = model.body(OBJECT).id
    return np.vstack(
        [
            geom_points(model, data, g)
            for g in range(model.ngeom)
            if model.body_rootid[model.geom_bodyid[g]] == body
            and (model.geom_contype[g] or model.geom_conaffinity[g])
        ]
    )


@dataclass
class Plan:
    """A pick: joint angles from above the object down to the grasp, then up."""

    path: List[np.ndarray]
    grasp_index: int
    place_x: float
    pitch: float
    width: float
    opening: float
    grip_z: float


def slab(points: np.ndarray, z: float) -> np.ndarray:
    """xy of the object's points within ``GRIP_SLAB`` of height ``z`` (wider if too few)."""
    for half in (GRIP_SLAB, 2 * GRIP_SLAB, np.inf):
        near = points[np.abs(points[:, 2] - z) <= half][:, :2]
        if len(near) >= 8:
            return near
    return points[:, :2]


def narrow_axis(band: np.ndarray) -> np.ndarray:
    """The footprint's narrow horizontal axis (PCA)."""
    _, vectors = np.linalg.eigh(np.cov((band - band.mean(axis=0)).T))
    return vectors[:, 0]


def solve_line(
    arm: Arm, targets: List[np.ndarray], pitch: float, closing: np.ndarray, start: np.ndarray
) -> Optional[List[np.ndarray]]:
    """Joint angles along tool positions 1 cm apart, from ``start``; None if the
    arm cannot follow them (out of reach, or a joint would jump)."""
    path = [start]
    for target in targets:
        q = arm.solve(target, pitch, closing, [path[-1]])
        if q is None or np.abs(q - path[-1]).max() > MAX_JOINT_JUMP:
            return None
        path.append(q)
    return path


def tilt_deg(quat0: np.ndarray, quat1: np.ndarray) -> float:
    """How far an object's up axis turned between two orientations (wxyz), degrees."""
    up0 = Rotation.from_quat(quat0[[1, 2, 3, 0]]).apply([0, 0, 1])
    up1 = Rotation.from_quat(quat1[[1, 2, 3, 0]]).apply([0, 0, 1])
    return float(np.degrees(np.arccos(np.clip(up0 @ up1, -1.0, 1.0))))


def plan_pick(
    arm: Arm,
    points: np.ndarray,
    grip_z: float,
    jaws: Jaws,
) -> Tuple[Optional[Plan], str]:
    """The steepest reachable pick of the object (``points`` with it at x = 0) at
    height ``grip_z`` (raised as far as the fingertips need above the table); or
    None and why."""
    top = points[:, 2].max()
    for place_x in PLACE_X:
        placed = points + [place_x, 0.0, 0.0]
        for pitch in PITCHES:
            # The fingertips reach GRIP_BEHIND_TIP past the grip point.
            down = np.sin(np.radians(pitch))
            z = max(grip_z, TABLE_Z + MIN_TIP_CLEARANCE + (GRIP_BEHIND_TIP + jaws.tip) * down)
            z = min(z, top)
            band = slab(placed, z)
            if pitch >= 90:
                narrow = narrow_axis(band)
                closings = [narrow, -narrow]
            else:
                # Tilted, the jaws close across the direction the arm reaches out in.
                out = band.mean(axis=0) - arm.pan_xy
                out /= np.linalg.norm(out)
                closings = [np.array([-out[1], out[0]]), np.array([out[1], -out[0]])]
            for closing in closings:
                across = band @ closing
                width = float(np.ptp(across))
                need = width + JAW_STANDOFF + OPEN_MARGIN
                # Fully open if nothing less will do: past ~70 mm the moving jaw has
                # swung away from the fingers' grasping length.
                opening = next(
                    (angle for angle, gap in jaws.clearances if gap >= need),
                    jaws.clearances[-1][0],
                )
                along = np.array([-closing[1], closing[0]])
                centre = (
                    closing * (across.max() + across.min()) / 2
                    + along * (np.ptp(band @ along) / 2 + (band @ along).min())
                )
                # The grip point, at the fixed jaw's face; the tool frame is off the
                # face towards the moving jaw (the tool's z is along closing).
                grip = np.array([*(centre - closing * (width / 2 + JAW_STANDOFF + jaws.face)), z])
                approach = np.array([0.0, 0.0, -1.0])
                # The tool's exact direction is the arm's to choose: solve, then refine.
                q = arm.solve(grip + approach * GRIP_BEHIND_TIP, pitch, closing, arm.seeds)
                if q is not None:
                    approach = arm.tcp(q)[1][:, 0]
                    q = arm.solve(grip + approach * GRIP_BEHIND_TIP, pitch, closing, [q])
                if q is None:
                    continue
                grasp = grip + approach * GRIP_BEHIND_TIP
                tip_z = grasp[2] + approach[2] * jaws.tip
                # Along the tool's axis from where the fingertips clear the top.
                reach = (top + APPROACH_CLEARANCE - tip_z) / max(-approach[2], 0.1)
                steps = np.arange(0.01, min(MAX_APPROACH, reach) + 1e-9, 0.01)
                approach_points = [grasp - approach * s for s in steps]
                lift_points = [grasp + [0, 0, s] for s in np.arange(0.01, LIFT + 1e-9, 0.01)]
                down_path = solve_line(arm, approach_points, pitch, closing, q)
                up_path = solve_line(arm, lift_points, pitch, closing, q) if down_path else None
                if not up_path:
                    continue
                return (
                    Plan(down_path[::-1] + up_path[1:], len(down_path) - 1, place_x, pitch,
                         width, opening, z),
                    "",
                )  # fmt: skip
    return None, "unreachable"


def run_trial(
    model: mujoco.MjModel,
    arm: Arm,
    rest: np.ndarray,
    mode: str,
    yaw: float,
    jaws: Jaws,
    renderer: Optional[mujoco.Renderer],
) -> Dict:
    """One pick of the object (``rest``: its free joint's qpos lying at x = 0),
    turned ``yaw`` degrees."""
    data = mujoco.MjData(model)
    obj_q = model.joint(model.body(OBJECT).jntadr[0]).qposadr[0]
    data.qpos[obj_q : obj_q + 7] = rest
    turn = np.array([np.cos(np.radians(yaw) / 2), 0, 0, np.sin(np.radians(yaw) / 2)])
    mujoco.mju_mulQuat(data.qpos[obj_q + 3 : obj_q + 7], turn, rest[3:])
    mujoco.mj_kinematics(model, data)
    points = object_vertices(model, data)
    top = points[:, 2].max()
    grip_z = top - GRIP_BELOW_TOP if mode == "top" else (top + TABLE_Z) / 2
    plan, reason = plan_pick(arm, points, grip_z, jaws)
    if plan is None:
        return {"outcome": reason}
    result = {
        "grip_mm": round((plan.grip_z - TABLE_Z) * 1000, 1),
        "width_mm": round(plan.width * 1000, 1),
        "place_x": plan.place_x,
        "pitch": plan.pitch,
    }

    arm_ctrl = np.array([model.actuator(name).id for name in ARM_JOINTS])
    gripper_ctrl = model.actuator("gripper").id
    gripper_q = model.joint("gripper").qposadr[0]
    data.qpos[obj_q] += plan.place_x
    data.qpos[arm.qadr] = plan.path[0]
    data.qpos[gripper_q] = plan.opening
    data.ctrl[arm_ctrl] = plan.path[0]
    data.ctrl[gripper_ctrl] = plan.opening
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    site = arm.site

    def run(seconds: float, ctrl=None) -> bool:
        steps = max(1, int(round(seconds / dt)))
        for i in range(steps):
            if ctrl is not None:
                ctrl((i + 1) / steps)
            mujoco.mj_step(model, data)
        return bool(np.all(np.isfinite(data.qpos)))

    def follow(qs: List[np.ndarray]) -> bool:
        """Plays joint angles 1 cm of tool travel apart at TOOL_SPEED."""
        stops = np.array(qs)

        def ctrl(s: float) -> None:
            k = s * (len(qs) - 1)
            i = min(int(k), len(qs) - 2)
            data.ctrl[arm_ctrl] = stops[i] + (stops[i + 1] - stops[i]) * (k - i)

        return run((len(qs) - 1) * 0.01 / TOOL_SPEED, ctrl)

    def object_pose() -> Tuple[np.ndarray, np.ndarray]:
        return data.qpos[obj_q : obj_q + 3].copy(), data.qpos[obj_q + 3 : obj_q + 7].copy()

    def in_tool(p: np.ndarray) -> np.ndarray:
        return data.site_xmat[site].reshape(3, 3).T @ (p - data.site_xpos[site])

    def close(s: float) -> None:
        data.ctrl[gripper_ctrl] = plan.opening + (GRIPPER_CLOSED - plan.opening) * min(1.0, 1.5 * s)

    ok = run(SETTLE_S)
    p0, q0 = object_pose()
    ok = ok and follow(plan.path[: plan.grasp_index + 1]) and run(0.2)
    p1, q1 = object_pose()
    knocked = np.linalg.norm(p1 - p0) > MAX_KNOCK or tilt_deg(q0, q1) > MAX_KNOCK_TILT
    ok = ok and run(CLOSE_S, close)
    gripped = in_tool(object_pose()[0])
    ok = ok and follow(plan.path[plan.grasp_index :]) and run(HOLD_S)
    if not ok:
        return {**result, "outcome": "unstable"}
    p2, _ = object_pose()
    rise = float(p2[2] - p0[2])
    slip = float(np.linalg.norm(in_tool(p2) - gripped))
    jaws = float(data.qpos[gripper_q])
    result.update(
        knock_mm=round(float(np.linalg.norm(p1 - p0)) * 1000, 1),
        rise_mm=round(rise * 1000, 1),
        slip_mm=round(slip * 1000, 1),
        jaws_rad=round(jaws, 3),
    )
    if renderer is not None:
        renderer.update_scene(data, camera="snapshot")
        result["image"] = renderer.render()

    # Whether it came up first: a thin object (a fork's handle) can be held with
    # the jaws all but closed (at their limit they are ~1 mm apart).
    if knocked:
        outcome = "knocked"
    elif rise >= LIFT_PASS * LIFT and slip <= MAX_SLIP:
        outcome = "ok"
    elif jaws < GRIPPER_CLOSED + 0.03:
        outcome = "missed"
    else:
        outcome = "slipped"
    return {**result, "outcome": outcome}


def test_object(args: Tuple[str, str, Optional[str]]) -> List[Dict]:
    """All grasps of one object; one row per grip height and turn."""
    source, objects_root, snapshots = args
    row = {"source": source}
    try:
        model = mujoco.MjModel.from_xml_string(model_xml(source, Path(objects_root)))
    except Exception as e:  # a broken asset: report it, carry on with the others
        return [{**row, "mode": "-", "outcome": f"load_error: {str(e).splitlines()[0][:80]}"}]
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    obj_q = model.joint(model.body(OBJECT).jntadr[0]).qposadr[0]
    points = object_vertices(model, data)
    lowest, size = points.min(axis=0), np.ptp(points, axis=0)
    rest = data.qpos[obj_q : obj_q + 7].copy()
    rest[2] += TABLE_Z + 0.001 - lowest[2]
    row.update(
        mass_g=round(float(model.body_subtreemass[model.body(OBJECT).id]) * 1000),
        size_mm=" x ".join(f"{v * 1000:.0f}" for v in size),
    )
    arm = Arm(model)
    jaws = measure_jaws(model)
    renderer = mujoco.Renderer(model, SNAPSHOT_SIZE[1], SNAPSHOT_SIZE[0]) if snapshots else None
    rows = []
    # Short objects: the middle is (nearly) the top.
    modes = ["top"] if size[2] / 2 - GRIP_BELOW_TOP < 0.01 else ["top", "middle"]
    for mode in modes:
        for yaw in YAWS:
            result = run_trial(model, arm, rest, mode, yaw, jaws, renderer)
            image = result.pop("image", None)
            if image is not None:
                import cv2

                name = f"{source.replace('/', '_')}_{mode}_{yaw}_{result['outcome']}.jpg"
                cv2.imwrite(str(Path(snapshots) / name), image[:, :, ::-1])
            rows.append({**row, "mode": mode, "yaw": yaw, **result})
    if renderer is not None:
        renderer.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--robocasa",
        type=Path,
        default=Path.home() / "robocasa",
        help="RoboCasa checkout, its object assets downloaded (default: ~/robocasa)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="<registry>/<category> to test, or <registry>/<category>/<instance>",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Where trials.csv (every grasp) and summary.csv (every object) go",
    )
    parser.add_argument(
        "--snapshots", action="store_true", help="Also save a picture of each grasp there"
    )
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    args = parser.parse_args()

    objects_root = args.robocasa.expanduser() / "robocasa" / "models" / "assets" / "objects"
    sources = []
    for category in args.categories:
        path = objects_root / category
        if (path / "model.xml").exists():
            sources.append(category)
        elif path.is_dir():
            sources += sorted(
                f"{category}/{d.name}" for d in path.iterdir() if (d / "model.xml").exists()
            )
        else:
            print(f"Skipping {category}: not in {objects_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshots = args.out_dir / "snapshots" if args.snapshots else None
    if snapshots:
        snapshots.mkdir(exist_ok=True)
        os.environ.setdefault("MUJOCO_GL", "egl")
    print(f"Testing {len(sources)} objects with {args.jobs} jobs", flush=True)

    started = time.monotonic()
    rows = []
    tasks = [(s, str(objects_root), str(snapshots) if snapshots else None) for s in sources]
    with multiprocessing.get_context("spawn").Pool(args.jobs) as pool:
        for i, result in enumerate(pool.imap_unordered(test_object, tasks), 1):
            rows += result
            ok = defaultdict(int)
            for r in result:
                ok[r["mode"]] += r["outcome"] == "ok"
            outcomes = ", ".join(f"{mode} {n}/{len(YAWS)}" for mode, n in ok.items())
            print(f"[{i}/{len(sources)}] {result[0]['source']}  {outcomes}", flush=True)
    rows.sort(key=lambda r: (r["source"], r["mode"], r.get("yaw", 0)))

    fields = [
        "source", "mode", "yaw", "outcome", "mass_g", "size_mm", "width_mm", "grip_mm",
        "place_x", "pitch", "knock_mm", "rise_mm", "slip_mm", "jaws_rad",
    ]  # fmt: skip
    with open(args.out_dir / "trials.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Per object: its best grip height, and what went wrong most at the other turns.
    by_object: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_object[r["source"]].append(r)
    summary = []
    for source, trials in sorted(by_object.items()):
        best_mode, best = "-", -1
        for mode in sorted({t["mode"] for t in trials}):
            n = sum(t["outcome"] == "ok" for t in trials if t["mode"] == mode)
            if n > best:
                best_mode, best = mode, n
        failures = [t["outcome"] for t in trials if t["mode"] == best_mode and t["outcome"] != "ok"]
        summary.append(
            {
                "source": source,
                "graspable": best >= MIN_SUCCESSES,
                "best_mode": best_mode,
                "successes": f"{max(best, 0)}/{len(YAWS)}",
                "failures": " ".join(sorted(set(failures))),
                "mass_g": trials[0].get("mass_g", ""),
                "size_mm": trials[0].get("size_mm", ""),
            }
        )
    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    graspable = defaultdict(list)
    tested = defaultdict(int)
    for s in summary:
        category, name = s["source"].rsplit("/", 1)
        tested[category] += 1
        if s["graspable"]:
            graspable[category].append(name)
    print(
        f"\nGraspable (>= {MIN_SUCCESSES}/{len(YAWS)} turns at one height), by category"
        f" ({time.monotonic() - started:.0f} s; details in {args.out_dir}):"
    )
    for category in sorted(tested):
        names = graspable[category]
        print(f"  {category}: {len(names)}/{tested[category]}  {' '.join(names)}")

if __name__ == "__main__":
    main()
