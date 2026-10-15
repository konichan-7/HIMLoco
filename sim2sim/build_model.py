"""Compile the pinned Unitree-derived URDF with local cylinder wheel collisions."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .config import JOINTS, MODEL, PHYSICS_DT, ROOT

SOURCE = ROOT / "legged_gym/resources/robots/go2w/urdf/go2w.urdf"


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
    # Keep primitive collision transforms at compiler precision, rather than
    # truncating the fused fixed-link transforms to saveLastXML's six digits.
    for body in root.findall(".//body"):
        compiled = imported.body(body.get("name"))
        indices = range(compiled.geomadr[0], compiled.geomadr[0] + compiled.geomnum[0])
        for geom, index in zip(body.findall("geom"), indices, strict=True):
            geom.set(
                "pos", " ".join(format(v, ".17g") for v in imported.geom_pos[index])
            )
            geom.set(
                "quat", " ".join(format(v, ".17g") for v in imported.geom_quat[index])
            )
    # MuJoCo 3.3.7 fixed-link fusion shifts the wheel-motor COM; restore the
    # explicit parallel-axis result. See docs/sim2sim_known_issues.md.
    for body in root.findall(".//body"):
        if body.get("name") not in inertias:
            continue
        mass, center, tensor = inertias[body.get("name")]
        inertial = body.find("inertial")
        inertial.attrib.clear()
        inertial.set("mass", format(mass, ".17g"))
        inertial.set("pos", " ".join(format(v, ".17g") for v in center))
        inertial.set(
            "fullinertia",
            " ".join(
                format(tensor[i, j], ".17g")
                for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
            ),
        )
    # Preserve URDF radians exactly instead of saveLastXML's six-digit ranges.
    for joint in root.findall(".//joint"):
        source_joint = urdf.find(f"joint[@name='{joint.get('name')}']")
        if source_joint is not None and source_joint.get("type") == "revolute":
            limit = source_joint.find("limit")
            joint.set("range", limit.get("lower") + " " + limit.get("upper"))
    root.find("compiler").set("inertiafromgeom", "false")
    ET.SubElement(
        root,
        "option",
        timestep=str(PHYSICS_DT),
        integrator="implicitfast",
        iterations="50",
        gravity="0 0 -9.81",
    )
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        friction="0.8 0 0",
        condim="3",
        contype="2",
        conaffinity="1",
        rgba="0.25 0.29 0.32 1",
    )
    ET.SubElement(world, "light", pos="0 -2 3", dir="0 0 -1")
    for body in world.iter("body"):
        for i, geom in enumerate(body.findall("geom")):
            geom.set("name", body.get("name") + f"_collision_{i}")
            geom.set("friction", "0.8 0 0")
            geom.set("condim", "3")
            geom.set("contype", "1")
            geom.set("conaffinity", "2")
            geom.set("rgba", "0.55 0.62 0.70 1")
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
            raise ValueError(f"{leg} wheel collision must be a cylinder along its joint axis")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=MODEL)
    parser.add_argument("--urdf", type=Path, default=SOURCE)
    args = parser.parse_args()
    print(build(args.output, args.urdf))
