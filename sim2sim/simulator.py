"""MuJoCo side of the Unitree LowCmd/LowState DDS connection."""

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from .build_model import initial_data
from .config import (
    COMMAND_TIMEOUT,
    JOINTS,
    LIMITS,
    MODEL,
    PHYSICS_DT,
    POLICY_TO_SDK,
    STATE_DT,
)
from .sdk import (
    ChannelPublisher,
    ChannelSubscriber,
    Latest,
    LowCmd_,
    LowState_,
    command_arrays,
    initialize,
    unitree_go_msg_dds__LowState_,
)


def motor_torques(values, q, dq):
    return np.clip(
        values[:, 2] * (values[:, 0] - q)
        + values[:, 3] * (values[:, 1] - dq)
        + values[:, 4],
        -LIMITS,
        LIMITS,
    )


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nu != 16 or model.nq != 23 or abs(model.opt.timestep - PHYSICS_DT) > 1e-9:
        raise ValueError(
            "Expected a floating Go2W with 16 motors and a 2 ms physics step"
        )
    data = initial_data(model)
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    vadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    aidx = np.array([model.actuator(name + "_motor").id for name in JOINTS])
    initialize(args.domain)
    command = Latest()
    bad_commands = 0

    def receive(msg):
        nonlocal bad_commands
        try:
            command.put(command_arrays(msg))
        except ValueError:
            bad_commands += 1

    subscriber = ChannelSubscriber("rt/lowcmd", LowCmd_)
    subscriber.Init(receive, 1)
    publisher = ChannelPublisher("rt/lowstate", LowState_)
    publisher.Init()

    def publish():
        msg = unitree_go_msg_dds__LowState_()
        msg.head = [0xFE, 0xEF]
        msg.tick = round(data.time * 1000)
        msg.imu_state.quaternion = data.sensor("imu_quat").data.tolist()
        msg.imu_state.gyroscope = data.sensor("imu_gyro").data.tolist()
        msg.imu_state.accelerometer = data.sensor("imu_accel").data.tolist()
        for i, sdk_index in enumerate(POLICY_TO_SDK):
            m = msg.motor_state[int(sdk_index)]
            m.mode = 1
            m.q, m.dq = float(data.qpos[qadr[i]]), float(data.qvel[vadr[i]])
            m.tau_est = float(data.actuator_force[aidx[i]])
        # Unsupported physical sensors remain zero; no fake battery/foot data.
        if not publisher.Write(msg):
            raise RuntimeError("LowState publication failed")

    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(model, data)
    frames = []
    watchdog_steps = 0
    saturation_steps = 0
    min_height = float(data.qpos[2])
    max_tilt = 0.0
    start = None
    loop_wall = 0.0
    try:
        with context as viewer:
            if viewer:
                viewer.cam.distance = 2.3
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -22
            deadline = time.monotonic() + 20
            while command.get()[0] is None:
                publish()
                if viewer:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                if time.monotonic() > deadline:
                    raise TimeoutError("No SDK LowCmd received within 20 seconds")
                time.sleep(0.01)
            start = time.monotonic()
            next_state = 0.0
            next_view = 0.0
            next_record = 0.0
            while args.duration == 0 or data.time < args.duration - 1e-9:
                if viewer and not viewer.is_running():
                    break
                now = time.monotonic()
                values, received_at, _ = command.get()
                if now - received_at > COMMAND_TIMEOUT:
                    # Simulation watchdog: damp all joints rather than applying stale targets.
                    torque = np.clip(-3 * data.qvel[vadr], -LIMITS, LIMITS)
                    watchdog_steps += 1
                else:
                    torque = motor_torques(values, data.qpos[qadr], data.qvel[vadr])
                saturation_steps += int(np.any(np.abs(torque) >= LIMITS - 1e-6))
                data.ctrl[aidx] = torque
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Nonfinite MuJoCo state")
                height = float(data.qpos[2])
                quat = data.qpos[3:7]
                tilt = float(
                    np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))
                )
                min_height, max_tilt = min(min_height, height), max(max_tilt, tilt)
                if data.time >= next_state - 1e-9:
                    publish()
                    next_state += STATE_DT
                if data.time >= next_record - 1e-9:
                    frames.append([data.time, *data.qpos[:7], *data.qvel[:6]])
                    next_record += 0.02
                if viewer and data.time >= next_view:
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                    next_view += 1 / 30
                delay = start + data.time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            loop_wall = time.monotonic() - start
            publish()
            if args.snapshot:
                with mujoco.Renderer(model, height=720, width=960) as renderer:
                    cam = mujoco.MjvCamera()
                    cam.lookat[:] = data.qpos[:3]
                    cam.distance, cam.azimuth, cam.elevation = 2.3, 135, -22
                    renderer.update_scene(data, camera=cam)
                    from PIL import Image

                    Image.fromarray(renderer.render()).save(args.snapshot)
            time.sleep(0.15)  # Deliver the final simulation tick before closing DDS.
    finally:
        subscriber.Close()
        publisher.Close()
        wall = loop_wall or (time.monotonic() - start if start else 0.0)
        report = {
            "simulated_seconds": data.time,
            "wall_seconds": wall,
            "total_wall_seconds": time.monotonic() - start if start else 0.0,
            "real_time_factor": data.time / wall if wall else 0.0,
            "received_commands": command.get()[2],
            "invalid_commands": bad_commands,
            "watchdog_steps": watchdog_steps,
            "saturation_steps": saturation_steps,
            "min_base_height_m": min_height,
            "max_tilt_deg": float(np.degrees(max_tilt)),
            "final_quaternion_wxyz": data.qpos[3:7].tolist(),
            "final_position_m": data.qpos[:3].tolist(),
            "model": str(args.model),
            "domain": args.domain,
            "state_publish_hz": 1 / STATE_DT,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if frames:
            np.savez_compressed(
                args.report.with_suffix(".npz"), trajectory=np.array(frames)
            )
        print("SIMULATOR", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--domain", type=int, default=71)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative (0 means until closed)")
    run(args)


if __name__ == "__main__":
    main()
