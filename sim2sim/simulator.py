"""MuJoCo side of the Unitree LowCmd/LowState DDS connection."""

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path

import mujoco
import numpy as np

from .config import (
    COMMAND_TIMEOUT,
    JOINTS,
    LIMITS,
    MODEL,
    NOMINAL,
    PHYSICS_DT,
    POLICY_DT,
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


def wait_for_policy_target(command, publish, tick):
    """Retry a boundary state if DDS drops it; never advance simulation time."""
    started = time.monotonic()
    next_publish = started
    publications = 0
    while True:
        now = time.monotonic()
        if now >= next_publish:
            publish()
            publications += 1
            next_publish = now + 0.01
        target = command.get()[0]
        if target is not None and target[1] == tick + 1:
            return time.monotonic() - started, publications - 1
        if now - started > 2.0:
            raise TimeoutError(f"No policy target acknowledging simulation tick {tick}")
        time.sleep(0.0002)


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    if model.nu != 16 or model.nq != 23 or abs(model.opt.timestep - PHYSICS_DT) > 1e-9:
        raise ValueError(
            "Expected a floating Go2W with 16 motors and a 2 ms physics step"
        )
    initial_height = 0.45  # ponytail: fixed Go2W startup height; tune here for another stance.
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, initial_height, 1, 0, 0, 0]
    for i, name in enumerate(JOINTS):
        data.qpos[model.joint(name).qposadr[0]] = NOMINAL[i] if i < 12 else 0.0
    mujoco.mj_forward(model, data)
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    vadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    aidx = np.array([model.actuator(name + "_motor").id for name in JOINTS])
    initialize(args.domain)
    command = Latest()
    bad_commands = 0

    def receive(msg):
        nonlocal bad_commands
        try:
            command.put((command_arrays(msg), int(msg.reserve)))
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
    watchdog_steps = 0
    saturation_steps = 0
    pd_updates = 0
    policy_barriers = 0
    barrier_wait_seconds = 0.0
    policy_state_retransmits = 0
    first_fall_time = None
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
            while args.duration == 0 or data.time < args.duration - 1e-9:
                if viewer and not viewer.is_running():
                    break
                # New targets every 20 ms, then ten independent 2 ms feedback
                # evaluations. A DDS acknowledgement prevents wall-clock drift
                # from changing the policy/PD ratio during offline evaluation.
                if pd_updates % round(POLICY_DT / PHYSICS_DT) == 0:
                    tick = round(data.time * 1000)
                    waited, retransmits = wait_for_policy_target(command, publish, tick)
                    barrier_wait_seconds += waited
                    policy_state_retransmits += retransmits
                    policy_barriers += 1
                now = time.monotonic()
                (values, _), received_at, _ = command.get()
                if now - received_at > COMMAND_TIMEOUT:
                    # Simulation watchdog: damp all joints rather than applying stale targets.
                    torque = np.clip(-3 * data.qvel[vadr], -LIMITS, LIMITS)
                    watchdog_steps += 1
                else:
                    torque = motor_torques(values, data.qpos[qadr], data.qvel[vadr])
                saturation_steps += int(np.any(np.abs(torque) >= LIMITS - 1e-6))
                data.ctrl[aidx] = torque
                pd_updates += 1
                mujoco.mj_step(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise RuntimeError("Nonfinite MuJoCo state")
                height = float(data.qpos[2])
                quat = data.qpos[3:7]
                tilt = float(
                    np.arccos(np.clip(1 - 2 * (quat[1] ** 2 + quat[2] ** 2), -1, 1))
                )
                min_height, max_tilt = min(min_height, height), max(max_tilt, tilt)
                if first_fall_time is None and (height < 0.20 or tilt > np.pi / 3):
                    first_fall_time = float(data.time)
                if data.time >= next_state - 1e-9:
                    publish()
                    next_state += STATE_DT
                if viewer and data.time >= next_view:
                    viewer.cam.lookat[:] = data.qpos[:3]
                    viewer.sync()
                    next_view += 1 / 30
                delay = start + data.time - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            loop_wall = time.monotonic() - start
            publish()
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
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "initial_height_m": initial_height,
            "domain": args.domain,
            "state_publish_hz": 1 / STATE_DT,
            "physics_hz": 1 / PHYSICS_DT,
            "pd_hz": pd_updates / data.time if data.time else 0,
            "pd_updates": pd_updates,
            "policy_barriers": policy_barriers,
            "pd_updates_per_policy": pd_updates / policy_barriers
            if policy_barriers
            else 0,
            "policy_barrier_wait_seconds": barrier_wait_seconds,
            "policy_state_retransmits": policy_state_retransmits,
            "timing_mode": "simulation_tick_lockstep",
            "first_fall_time_s": first_fall_time,
            "fall_definition": "base height < 0.20 m or tilt > 60 degrees",
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("SIMULATOR", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--domain", type=int, default=71)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative (0 means until closed)")
    run(args)


if __name__ == "__main__":
    main()
