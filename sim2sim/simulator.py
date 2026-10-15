"""MuJoCo side of the Unitree LowCmd/LowState DDS connection."""

import argparse
import hashlib
import json
import struct
import time
from contextlib import nullcontext
from pathlib import Path

import glfw
import mujoco
import numpy as np

from .config import (
    COMMAND_TIMEOUT,
    JOINTS,
    KEYBOARD_SPEED,
    KEYBOARD_TURN_RATE,
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


class KeyboardInput:
    """Hold/release control inside the native viewer, without a second window."""

    def __init__(self, speed, turn_rate):
        self.speed, self.turn_rate = speed, turn_rate
        self.held = set()
        self.window = None
        self.viewer = None

    def in_viewport(self, window):
        x, y = glfw.get_cursor_pos(window)
        width, height = glfw.get_window_size(window)
        fw, fh = glfw.get_framebuffer_size(window)
        rect = self.viewer.viewport if self.viewer else None
        if not rect or not width or not height:
            return False
        x, y = x * fw / width, (height - y) * fh / height
        return (
            rect.left <= x < rect.left + rect.width
            and rect.bottom <= y < rect.bottom + rect.height
        )

    def bootstrap(self, key):
        # MuJoCo 3.3.7 exposes presses only. Its UI thread owns this GLFW context.
        # Chain the native C callback: glfw.set_key_callback loses that pointer.
        if self.window is not None:
            return
        self.window = glfw.get_current_context()
        if not self.window:
            raise RuntimeError("Native viewer has no current GLFW context")
        # ponytail: bridge pinned GLFW callbacks until the viewer exposes releases.
        setter = glfw._glfw.glfwSetKeyCallback
        self.callback = setter.restype(self.key_event)
        self.native_callback = setter(self.window, self.callback)
        if glfw.KEY_1 <= key <= glfw.KEY_4 and self.in_viewport(self.window):
            # Native UI already consumed the first digit: undo its geom-group toggle.
            self.native_callback(self.window, key, 0, glfw.PRESS, 0)
            self.key_event(self.window, key, 0, glfw.PRESS, 0)
        elif key == glfw.KEY_ESCAPE:
            self.key_event(self.window, key, 0, glfw.PRESS, 0)

    def key_event(self, window, key, scancode, action, mods):
        if glfw.KEY_1 <= key <= glfw.KEY_4:
            if action == glfw.RELEASE:
                self.held.discard(key)
                return
            if key in self.held:
                return
            if action == glfw.PRESS and not mods and self.in_viewport(window):
                self.held.add(key)
                return
        if key == glfw.KEY_ESCAPE:
            self.held.clear()
            glfw.set_window_should_close(window, True)
            return
        self.native_callback(window, key, scancode, action, mods)

    def command(self):
        # GLFW synthesizes releases for held keys when the window loses focus.
        held = self.held.copy()
        return (
            self.speed * ((glfw.KEY_1 in held) - (glfw.KEY_2 in held)),
            0.0,
            self.turn_rate * ((glfw.KEY_3 in held) - (glfw.KEY_4 in held)),
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
            active = publish()
            publications += 1
            if active is False:
                return time.monotonic() - started, publications - 1
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
    initial_height = 0.45  # Go2W startup clearance above terrain.
    decimation = round(POLICY_DT / PHYSICS_DT)
    qadr = np.array([model.joint(name).qposadr[0] for name in JOINTS])
    vadr = np.array([model.joint(name).dofadr[0] for name in JOINTS])
    aidx = np.array([model.actuator(name + "_motor").id for name in JOINTS])
    data = mujoco.MjData(model)
    data.qpos[:3] = args.spawn
    data.qpos[2] += initial_height
    data.qpos[3:7] = [1, 0, 0, 0]
    data.qpos[qadr] = np.r_[NOMINAL, np.zeros(4)]
    mujoco.mj_forward(model, data)
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
        if args.keyboard:
            # Local simulation protocol: three float32 velocity commands.
            msg.wireless_remote = (
                list(struct.pack("<3f", *keyboard.command())) + [0] * 28
            )
        msg.wireless_remote[12] = int(viewer is not None and not viewer.is_running())
        for i, sdk_index in enumerate(POLICY_TO_SDK):
            m = msg.motor_state[int(sdk_index)]
            m.mode = 1
            m.q, m.dq = float(data.qpos[qadr[i]]), float(data.qvel[vadr[i]])
            m.tau_est = float(data.actuator_force[aidx[i]])
        # Unsupported physical sensors remain zero; no fake battery/foot data.
        if not publisher.Write(msg):
            raise RuntimeError("LowState publication failed")
        return viewer.is_running() if viewer else True

    keyboard = KeyboardInput(args.speed, args.turn_rate) if args.keyboard else None
    if args.headless:
        context = nullcontext(None)
    else:
        from mujoco import viewer as mj_viewer

        context = mj_viewer.launch_passive(
            model, data, key_callback=keyboard.bootstrap if keyboard else None
        )
    if keyboard:
        print(
            "KEYBOARD: mouse over scene; hold 1 forward, 2 reverse, 3 left, 4 right; "
            "release/unfocus = zero. Native UI: Rendering > Contact point / Contact force",
            flush=True,
        )
    watchdog_steps = 0
    saturation_steps = 0
    pd_updates = 0
    policy_barriers = 0
    barrier_wait_seconds = 0.0
    policy_state_retransmits = 0
    first_fall_time = None
    min_height = initial_height
    clearance = initial_height
    ground_groups = np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    max_tilt = 0.0
    start = None
    loop_wall = 0.0
    try:
        with context as viewer:
            if viewer:
                if keyboard:
                    keyboard.viewer = viewer
                with viewer.lock():
                    viewer.opt.geomgroup[3] = 0
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1
                    viewer.cam.distance = 2.3
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -22
                    viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()  # Compile renderer shaders before publishing the first state.
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
                if pd_updates % decimation == 0:
                    tick = round(data.time * 1000)
                    waited, retransmits = wait_for_policy_target(command, publish, tick)
                    if viewer and not viewer.is_running():
                        break
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
                # Height relative to terrain, so elevated stairs do not hide falls.
                if pd_updates % decimation == 0:
                    distance = mujoco.mj_ray(
                        model,
                        data,
                        data.qpos[:3],
                        np.array([0.0, 0.0, -1.0]),
                        ground_groups,
                        1,
                        -1,
                        np.empty(1, dtype=np.int32),
                    )
                    clearance = float(distance) if distance >= 0 else 0.0
                height = clearance
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
                    with viewer.lock():
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
            "spawn_ground_xyz": args.spawn,
            "height_reference": "terrain directly below base",
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
            "fall_definition": "base clearance above terrain < 0.20 m or tilt > 60 degrees",
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print("SIMULATOR", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument(
        "--spawn",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Ground xyz at spawn",
    )
    parser.add_argument("--domain", type=int, default=71)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--speed", type=float, default=KEYBOARD_SPEED)
    parser.add_argument("--turn-rate", type=float, default=KEYBOARD_TURN_RATE)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error("duration must be finite and nonnegative (0 means until closed)")
    if args.keyboard and args.headless:
        parser.error("--keyboard requires a visible window")
    if not all(np.isfinite(v) and v > 0 for v in (args.speed, args.turn_rate)):
        parser.error("--speed and --turn-rate must be finite and positive")
    run(args)


if __name__ == "__main__":
    main()
