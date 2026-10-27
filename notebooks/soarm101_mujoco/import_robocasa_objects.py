"""Copy kitchen objects from RoboCasa onto the sim's table.

RoboCasa (https://github.com/robocasa/robocasa) ships its objects as one MJCF
model each (``objects/<registry>/<category>/<instance>/model.xml``: a textured
visual mesh plus a convex decomposition for collisions), downloaded with
``python -m robocasa.scripts.download_kitchen_assets``. This script copies the
ones in ``OBJECTS`` into ``SO101/assets/objects/`` and writes
``SO101/objects.xml`` (included by ``scene.xml``): every object a free body
resting on the table top, its visual geoms in group 2 and its collision geoms in
group 3 (hidden by default), like the arm's.

Run (with mujoco installed; RoboCasa's assets downloaded)::

    python3 notebooks/soarm101_mujoco/import_robocasa_objects.py
    python3 notebooks/soarm101_mujoco/import_robocasa_objects.py --robocasa ~/robocasa
"""

import argparse
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import mujoco
import numpy as np

SO101_DIR = Path(__file__).resolve().parent / "SO101"
OUT_ASSETS = SO101_DIR / "assets" / "objects"
OUT_XML = SO101_DIR / "objects.xml"
# Table top (scene.xml), metres; objects are dropped this far above it.
TABLE_Z = -0.0024
DROP_GAP = 0.002
COLLISION_DENSITY = 500.0  # kg/m^3: the convex hulls fill hollow objects


@dataclass
class TabletopObject:
    """One RoboCasa object instance and where it goes.

    Args:
        name: Body name in the scene (also the asset prefix).
        source: ``<registry>/<category>/<instance>`` under RoboCasa's objects.
        scale: Uniform scale on top of the model's own (1 = as RoboCasa ships it).
        x, y: Position on the table, metres (the arm's base at the origin, x forward).
        yaw: Rotation about z, degrees.
    """

    name: str
    source: str
    scale: float
    x: float
    y: float
    yaw: float = 0.0


# Within the arm's reach (~0.4 m), around the calibration sheets' middle.
OBJECTS: List[TabletopObject] = [
    TabletopObject("mug", "objaverse/mug/mug_1", 1.0, 0.22, 0.16, 200),
    TabletopObject("cup", "objaverse/cup/cup_2", 1.0, 0.30, 0.06),
    TabletopObject("bowl", "objaverse/bowl/bowl_0", 1.0, 0.33, -0.12),
    TabletopObject("plate", "objaverse/plate/plate_0", 1.0, 0.45, 0.10),
    TabletopObject("spoon", "objaverse/spoon/spoon_0", 1.0, 0.18, -0.20, 20),
    TabletopObject("fork", "objaverse/fork/fork_0", 1.0, 0.24, -0.26, 10),
    TabletopObject("banana", "objaverse/banana/banana_1", 0.95, 0.20, 0.27, 30),
    TabletopObject("apple", "objaverse/apple/apple_0", 0.9, 0.36, 0.24),
    TabletopObject("milk", "objaverse/milk/milk_0", 1.0, 0.42, -0.28),
    TabletopObject("water_bottle", "objaverse/bottled_water/bottled_water_1", 1.0, 0.48, -0.10),
]

# Attributes naming an asset, by element.
REFS = {"geom": ("mesh", "material"), "material": ("texture",)}


def geom_kind(geom: ET.Element) -> str:
    """"visual", "collision" or "skip" for a geom of a RoboCasa object.

    Visual geoms are in group 1 (objaverse sets it on the geom, lightwheel through
    the "visual" class); collision ones in group 0 (some via the "collision"
    class); "region" and "spawn" geoms only guide RoboCasa's placement.
    """
    if geom.get("class") in ("region", "spawn") or geom.get("type") != "mesh":
        return "skip"
    if geom.get("group") == "1" or geom.get("class") == "visual":
        return "visual"
    return "collision"


def import_object(
    obj: TabletopObject, objects_root: Path, asset: ET.Element, copy: bool = True
) -> ET.Element:
    """Adds one object's assets and returns its (free) body.

    Args:
        copy: Copy the object's files into ``SO101/assets/objects/``; otherwise
            the assets refer to RoboCasa's files by absolute path (for models
            that are only compiled, e.g. by ``grasp_test.py``).
    """
    src_dir = objects_root / obj.source
    model = ET.parse(src_dir / "model.xml").getroot()
    dst_dir = OUT_ASSETS / obj.name
    if copy and dst_dir.exists():
        shutil.rmtree(dst_dir)

    renamed: Dict[str, str] = {}
    for el in model.find("asset"):
        old = el.get("name")
        renamed[old] = f"{obj.name}_{old}"
        el.set("name", renamed[old])
        file = el.get("file")
        if file and copy:
            (dst_dir / file).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dir / file, dst_dir / file)
            # meshdir and texturedir are both "assets".
            el.set("file", f"objects/{obj.name}/{file}")
        elif file:
            el.set("file", str((src_dir / file).resolve()))
        if el.tag == "mesh":
            scale = np.array([float(v) for v in el.get("scale", "1 1 1").split()]) * obj.scale
            el.set("scale", " ".join(f"{v:.6g}" for v in scale))
        for attr in REFS.get(el.tag, ()):
            if el.get(attr):
                el.set(attr, renamed[el.get(attr)])
        asset.append(el)

    source_body = model.find(".//body[@name='object']")
    body = ET.Element("body", name=obj.name)
    ET.SubElement(body, "freejoint", name=f"{obj.name}_joint")
    inner = ET.SubElement(body, "body", name=f"{obj.name}_model")
    for attr in ("pos", "quat"):
        if source_body.get(attr):
            inner.set(attr, source_body.get(attr))
    if source_body.get("pos"):
        pos = np.array([float(v) for v in source_body.get("pos").split()]) * obj.scale
        inner.set("pos", " ".join(f"{v:.6g}" for v in pos))

    for geom in source_body.iter("geom"):
        kind = geom_kind(geom)
        if kind == "skip":
            continue
        out = ET.SubElement(inner, "geom", type="mesh", mesh=renamed[geom.get("mesh")])
        if geom.get("pos"):
            pos = np.array([float(v) for v in geom.get("pos").split()]) * obj.scale
            out.set("pos", " ".join(f"{v:.6g}" for v in pos))
        if geom.get("quat"):
            out.set("quat", geom.get("quat"))
        if kind == "visual":
            out.set("group", "2")
            out.set("contype", "0")
            out.set("conaffinity", "0")
            out.set("mass", "0")
            if geom.get("material"):
                out.set("material", renamed[geom.get("material")])
        else:
            out.set("group", "3")
            out.set("density", f"{COLLISION_DENSITY:g}")
            out.set("friction", "1 0.01 0.001")
    return body


def bottom_offsets(xml_path: Path, names: List[str]) -> Dict[str, float]:
    """Height of each body's origin above its lowest collision vertex, metres."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    offsets = {}
    for name in names:
        body_id = model.body(name).id
        lowest = np.inf
        for g in range(model.ngeom):
            root = model.geom_bodyid[g]
            while root and model.body_parentid[root] != 0:
                root = model.body_parentid[root]
            if root != body_id or model.geom_group[g] != 3:
                continue
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            verts = model.mesh_vert[start : start + count]
            world = data.geom_xpos[g] + verts @ data.geom_xmat[g].reshape(3, 3).T
            lowest = min(lowest, world[:, 2].min())
        offsets[name] = float(data.xpos[body_id][2] - lowest)
    return offsets


def write_objects_xml(objects_root: Path, heights: Dict[str, float]) -> None:
    root = ET.Element("mujoco", model="robocasa_objects")
    asset = ET.SubElement(root, "asset")
    worldbody = ET.SubElement(root, "worldbody")
    for obj in OBJECTS:
        body = import_object(obj, objects_root, asset)
        z = TABLE_Z + DROP_GAP + heights.get(obj.name, 0.0)
        body.set("pos", f"{obj.x:g} {obj.y:g} {z:.5f}")
        yaw = np.radians(obj.yaw) / 2
        body.set("quat", f"{np.cos(yaw):.6g} 0 0 {np.sin(yaw):.6g}")
        worldbody.append(body)
    ET.indent(root, space="    ")
    comment = (
        "Generated by import_robocasa_objects.py from RoboCasa's objects (CC BY 4.0);"
        " do not edit."
    )
    OUT_XML.write_text(f"<!-- {comment} -->\n" + ET.tostring(root, encoding="unicode") + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--robocasa",
        type=Path,
        default=Path.home() / "robocasa",
        help="RoboCasa checkout, its object assets downloaded (default: ~/robocasa)",
    )
    args = parser.parse_args()
    objects_root = args.robocasa.expanduser() / "robocasa" / "models" / "assets" / "objects"

    # First pass at height 0 to measure each object's bottom, then the real one.
    write_objects_xml(objects_root, {})
    heights = bottom_offsets(SO101_DIR / "scene.xml", [obj.name for obj in OBJECTS])
    write_objects_xml(objects_root, heights)
    size = sum(f.stat().st_size for f in OUT_ASSETS.rglob("*") if f.is_file())
    print(f"Wrote {OUT_XML} ({len(OBJECTS)} objects, {size / 1e6:.1f} MB of assets)")


if __name__ == "__main__":
    main()
