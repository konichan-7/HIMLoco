"""Compile the pinned Unitree-derived URDF with local cylinder wheel collisions."""

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .config import JOINTS, MODEL, PHYSICS_DT, ROOT

SOURCE = ROOT / "legged_gym/resources/robots/go2w/urdf/go2w.urdf"


def numbers(values):
    return " ".join(format(v, ".17g") for v in values)


def combined_inertias(urdf):
    """Combine explicit fixed-link inertias in each moving link's frame."""
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    groups = {}

    def transform(element):
        if element is None:
            return np.eye(4)
        angles = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        cr, cp, cy = np.cos(angles)
        sr, sp, sy = np.sin(angles)
        matrix = np.eye(4)
        matrix[:3, :3] = [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
        matrix[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        return matrix

    def walk(name, group, frame):
        inertial = links[name].find("inertial")
        if inertial is not None:
            mass = float(inertial.find("mass").get("value"))
            if mass > 0:
                pose = frame @ transform(inertial.find("origin"))
                values = inertial.find("inertia").attrib
                xx, yy, zz, xy, xz, yz = (
                    float(values[k]) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                )
                tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
                rotation = pose[:3, :3]
                groups.setdefault(group, []).append(
                    (mass, pose[:3, 3], rotation @ tensor @ rotation.T)
                )
        for joint in children.get(name, []):
            child = joint.find("child").get("link")
            fixed = joint.get("type") == "fixed"
            walk(
                child,
                group if fixed else child,
                frame @ transform(joint.find("origin")) if fixed else np.eye(4),
            )

    walk("base", "base", np.eye(4))
    combined = {}
    for name, parts in groups.items():
        mass = sum(p[0] for p in parts)
        center = sum(m * p for m, p, _ in parts) / mass
        tensor = np.zeros((3, 3))
        for m, p, inertia in parts:
            d = p - center
            tensor += inertia + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        combined[name] = mass, center, tensor
    return combined


def add_visuals(root, source, urdf, output):
    """Share source STL assets; convert only unsupported DAE material parts."""
    import trimesh

    asset = ET.SubElement(root, "asset")
    meshes = {}
    for link in urdf.findall("link"):
        body = root.find(f".//body[@name='{link.get('name')}']")
        for index, visual in enumerate(link.findall("visual")):
            if body is None:
                raise ValueError(f"Missing visual body: {link.get('name')}")
            mesh = visual.find("geometry/mesh")
            path = (source.parent / mesh.get("filename")).resolve()
            # Pinned Gym-only DAE wrappers reproduce these source STLs exactly.
            if path.name in ("calf.dae", "calf_mirror.dae"):
                path = path.with_suffix(".stl")
            if path not in meshes:
                fallback = visual.find("material/color")
                rgba = (
                    fallback.get("rgba") if fallback is not None else "0.15 0.15 0.15 1"
                )
                parts = []
                if path.suffix.lower() in (".stl", ".obj", ".msh"):
                    name = f"visual_{path.stem}_0"
                    ET.SubElement(
                        asset,
                        "mesh",
                        name=name,
                        file=os.path.relpath(path, output.parent),
                        scale=mesh.get("scale", "1 1 1"),
                    )
                    parts.append((name, rgba))
                else:
                    scene = trimesh.load(path, force="scene", process=False)
                    for part, node in enumerate(scene.graph.nodes_geometry):
                        transform, geometry = scene.graph[node]
                        shape = scene.geometry[geometry].copy()
                        shape.apply_transform(transform)
                        target = (
                            output.parent.parent
                            / "meshes/mujoco"
                            / f"{path.stem}_{part}.obj"
                        )
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shape.export(target, include_color=False, include_texture=False)
                        name = f"visual_{path.stem}_{part}"
                        ET.SubElement(
                            asset,
                            "mesh",
                            name=name,
                            file=os.path.relpath(target, output.parent),
                            scale=mesh.get("scale", "1 1 1"),
                        )
                        material = getattr(shape.visual, "material", None)
                        color = getattr(material, "baseColorFactor", None)
                        rgba = (
                            " ".join(str(float(v) / 255) for v in color)
                            if color is not None
                            else fallback.get("rgba")
                            if fallback is not None
                            else "0.15 0.15 0.15 1"
                        )
                        parts.append((name, rgba))
                meshes[path] = parts
            origin = visual.find("origin")
            for name, rgba in meshes[path]:
                ET.SubElement(
                    body,
                    "geom",
                    name=f"{link.get('name')}_visual_{index}_{name}",
                    type="mesh",
                    mesh=name,
                    pos=origin.get("xyz", "0 0 0"),
                    euler=origin.get("rpy", "0 0 0"),
                    rgba=rgba,
                    group="1",
                    contype="0",
                    conaffinity="0",
                    mass="0",
                )
    asset.extend(
        ET.fromstring("""<asset>
      <texture name="ground_grid" type="2d" builtin="checker"
               rgb1="0.20 0.24 0.28" rgb2="0.28 0.33 0.37" width="512" height="512"/>
      <material name="ground" texture="ground_grid" texrepeat="8 8"
                texuniform="true" reflectance="0"/>
      <texture name="sky" type="skybox" builtin="gradient"
               rgb1="0.30 0.45 0.60" rgb2="0.06 0.09 0.13" width="512" height="3072"/>
    </asset>""")
    )
    root.find("worldbody/geom[@name='floor']").set("material", "ground")
    root.append(
        ET.fromstring("""<visual>
      <headlight ambient="0.2 0.2 0.2" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
      <global offwidth="1600" offheight="1000"/>
      <quality shadowsize="2048" offsamples="4"/>
    </visual>""")
    )


def build(output=MODEL, source=SOURCE):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    urdf = ET.parse(source).getroot()
    if urdf.findall("link/collision/geometry/mesh"):
        raise ValueError("This Go2W builder supports primitive collisions only")
    inertias = combined_inertias(urdf)
    torque_limits = {
        n: float(urdf.find(f"joint[@name='{n}']/limit").get("effort")) for n in JOINTS
    }
    visual_urdf = ET.fromstring(ET.tostring(urdf))
    for link in urdf.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)
    # Without an explicit floating parent, a URDF importer can fuse base inertia
    # into the fixed world. Explicit inertias must never be inferred from geoms.
    ET.SubElement(urdf, "link", name="world")
    floating = ET.SubElement(urdf, "joint", name="floating_base", type="floating")
    ET.SubElement(floating, "parent", link="world")
    ET.SubElement(floating, "child", link="base")
    extension = ET.SubElement(urdf, "mujoco")
    ET.SubElement(
        extension,
        "compiler",
        fusestatic="true",
        inertiafromgeom="false",
    )
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with tempfile.TemporaryDirectory() as temporary:
        imported_path = Path(temporary) / "imported.xml"
        mujoco.mj_saveLastXML(str(imported_path), imported)
        root = ET.parse(imported_path).getroot()
    # One pass restores compiler-precision collisions and explicit fixed-link
    # inertias (MuJoCo's fixed-link fusion shifts the wheel-motor COM).
    for body in root.findall(".//body"):
        name = body.get("name")
        compiled = imported.body(name)
        indices = range(compiled.geomadr[0], compiled.geomadr[0] + compiled.geomnum[0])
        for i, (geom, index) in enumerate(
            zip(body.findall("geom"), indices, strict=True)
        ):
            geom.attrib.update(
                name=f"{name}_collision_{i}",
                pos=numbers(imported.geom_pos[index]),
                quat=numbers(imported.geom_quat[index]),
                group="3",
                friction="0.8 0 0",
                condim="3",
                contype="1",
                conaffinity="2",
                rgba="0.55 0.62 0.70 1",
            )
        if name in inertias:
            mass, center, tensor = inertias[name]
            body.find("inertial").attrib = {
                "mass": format(mass, ".17g"),
                "pos": numbers(center),
                "fullinertia": numbers(
                    tensor[i, j]
                    for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
                ),
            }
    # Preserve URDF radians exactly instead of saveLastXML's six-digit ranges.
    for joint in root.findall(".//joint"):
        source_joint = urdf.find(f"joint[@name='{joint.get('name')}']")
        if source_joint is not None and source_joint.get("type") == "revolute":
            limit = source_joint.find("limit")
            joint.set("range", limit.get("lower") + " " + limit.get("upper"))
    root.find("compiler").set("inertiafromgeom", "false")
    root.append(
        ET.fromstring(f"""<option timestep="{PHYSICS_DT}"
        integrator="implicitfast" iterations="50" gravity="0 0 -9.81"/>""")
    )
    world = root.find("worldbody")
    world.append(
        ET.fromstring("""<geom name="floor" type="plane" size="0 0 0.1"
        friction="0.8 0 0" condim="3" contype="2" conaffinity="1"
        rgba="0.25 0.29 0.32 1"/>""")
    )
    world.find("body[@name='base']").append(
        ET.fromstring("""<light name="key_light"
        ambient="0.1 0.1 0.1" diffuse="0.5 0.5 0.5" specular="0.1 0.1 0.1"
        mode="trackcom" directional="true" pos="0 -2 3" dir="0.2 0.4 -1"/>""")
    )
    # These are scene choices, not undocumented properties of the URDF.
    base = world.find("body[@name='base']")
    imu_joint = urdf.find("joint[@name='imu_joint']")
    if imu_joint.find("parent").get("link") != "base":
        raise ValueError("Expected the pinned URDF IMU frame to be fixed to base")
    imu_origin = imu_joint.find("origin")
    ET.SubElement(
        base,
        "site",
        name="imu",
        pos=imu_origin.get("xyz"),
        euler=imu_origin.get("rpy"),
        size="0.005",
    )
    actuators = ET.SubElement(root, "actuator")
    for name in JOINTS:
        limit = torque_limits[name]
        ET.SubElement(
            actuators,
            "motor",
            name=name + "_motor",
            joint=name,
            ctrlrange=f"{-limit} {limit}",
            ctrllimited="true",
        )
    sensors = ET.SubElement(root, "sensor")
    ET.SubElement(sensors, "framequat", name="imu_quat", objtype="site", objname="imu")
    ET.SubElement(sensors, "gyro", name="imu_gyro", site="imu")
    ET.SubElement(sensors, "accelerometer", name="imu_accel", site="imu")
    add_visuals(root, source, visual_urdf, output)
    ET.indent(root)
    ET.ElementTree(root).write(output, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(output))
    # Keep one build-time check; historical failure cases are recorded in docs.
    expected_mass = sum(values[0] for values in inertias.values())
    if (model.nq, model.nv, model.nu) != (23, 22, 16) or not np.isclose(
        model.body_mass.sum(), expected_mass, atol=1e-8, rtol=0
    ):
        raise ValueError("URDF conversion changed floating-base dimensions or mass")
    for leg in ("FL", "FR", "RL", "RR"):
        geom = model.geom(leg + "_foot_collision_0")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, geom.quat)
        axle = rotation.reshape(3, 3)[:, 2]
        joint_axis = model.joint(leg + "_foot_joint").axis
        if geom.type[0] != mujoco.mjtGeom.mjGEOM_CYLINDER or not np.isclose(
            abs(axle @ joint_axis), 1.0, atol=1e-10
        ):
            raise ValueError(
                f"{leg} wheel collision must be a cylinder along its joint axis"
            )
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL)
    parser.add_argument("--urdf", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(build(args.output, args.urdf))
