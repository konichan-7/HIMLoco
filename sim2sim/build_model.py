"""Adapt Unitree's upstream Go2W MJCF to the local DDS sim2sim contract."""

import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from .config import JOINTS, LIMITS, MODEL, NOMINAL, PHYSICS_DT, ROOT

RESOURCES = ROOT / "legged_gym/resources/robots/go2w"
SOURCE = RESOURCES / "xml/go2w.xml"
ASSETS = RESOURCES / "meshes"
SOURCE_COMMIT = "1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d"


def build():
    """Build the runtime model without modifying the vendored upstream inputs."""
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    mjcf = ET.parse(SOURCE).getroot()
    compiler = mjcf.find("compiler")
    compiler.set(
        "meshdir",
        os.path.relpath(ASSETS, MODEL.parent),
    )

    # The upstream MJCF calls these wheel joints; the policy/SDK contract in
    # this repository uses the robot's LowState names (foot_joint).
    for element in mjcf.iter():
        if element.tag == "joint" and element.get("name", "").endswith(
            "_wheel_joint"
        ):
            element.set(
                "name", element.get("name").replace("_wheel_joint", "_foot_joint")
            )
        if element.get("joint", "").endswith("_wheel_joint"):
            element.set(
                "joint", element.get("joint").replace("_wheel_joint", "_foot_joint")
            )

    option = mjcf.find("option")
    option.set("timestep", str(PHYSICS_DT))
    option.set("integrator", "implicitfast")
    option.set("iterations", "50")

    # Use policy-order actuator names and the same torque limits as the
    # LowCmd feedback model. Upstream actuator order follows SDK order instead.
    limits = dict(zip(JOINTS, LIMITS))
    for actuator in mjcf.findall("actuator/*"):
        joint = actuator.get("joint")
        actuator.set("name", joint + "_motor")
        limit = float(limits[joint])
        actuator.set("ctrllimited", "true")
        actuator.set("ctrlrange", f"{-limit:g} {limit:g}")

    for sensor in mjcf.findall("sensor/*"):
        if sensor.get("name") == "imu_acc":
            sensor.set("name", "imu_accel")

    world = mjcf.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="0 0 0.1",
        rgba=".25 .29 .32 1",
        friction=".8 .005 .0001",
        condim="6",
    )
    ET.SubElement(
        world,
        "light",
        name="key_light",
        pos="0 -2 3",
        dir="0 0 -1",
        diffuse=".8 .8 .8",
    )
    ET.indent(mjcf)
    ET.ElementTree(mjcf).write(MODEL, encoding="unicode", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(MODEL))
    if model.nu != 16 or model.nq != 23 or model.nv != 22:
        raise RuntimeError(
            f"Unexpected upstream Go2W dimensions nq/nv/nu="
            f"{model.nq}/{model.nv}/{model.nu}"
        )
    info = {
        "source_repository": "https://github.com/unitreerobotics/unitree_mujoco",
        "source_commit": SOURCE_COMMIT,
        "source_path": "unitree_robots/go2w/go2w.xml",
        "local_source_path": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "asset_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ASSETS.iterdir())
            if path.name in {mesh.get("file") for mesh in mjcf.findall("asset/mesh")}
        },
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "mass_kg": float(model.body_mass.sum()),
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "joint_names": list(JOINTS),
        "actuator_names": [model.actuator(i).name for i in range(model.nu)],
        "changes": [
            "adapt upstream actuator names to policy order and framework limits",
            "set the local 2 ms timestep and implicitfast integrator",
            "add the framework ground plane and key light",
            "rename the upstream IMU accelerometer sensor to imu_accel",
        ],
    }
    (MODEL.parent / "model_manifest.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )
    print(json.dumps(info, indent=2))
    return MODEL


def initial_data(model):
    data = mujoco.MjData(model)
    data.qpos[:7] = [0.0, 0.0, 0.6, 1.0, 0.0, 0.0, 0.0]
    for i, name in enumerate(JOINTS):
        joint = model.joint(name)
        data.qpos[joint.qposadr[0]] = NOMINAL[i] if i < 12 else 0.0
    mujoco.mj_forward(model, data)
    return data


if __name__ == "__main__":
    build()
