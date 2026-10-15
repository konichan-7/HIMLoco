"""TorchScript inference for the 53- and 57-dimensional Go2W policy layouts."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .config import HISTORY, JOINTS, LIMITS, POLICY_TO_SDK


def projected_gravity(quaternion):
    q = np.asarray(quaternion, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-6:
        raise ValueError("Invalid wxyz quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1],
        dtype=np.float32,
    )


class Policy:
    def __init__(self, path, device="auto", metadata_path=None):
        self.path = Path(path).resolve()
        self.cfg = json.loads(
            Path(metadata_path or self.path.with_name("metadata.json")).read_text()
        )
        if (
            self.cfg.get("policy_sha256")
            and hashlib.sha256(self.path.read_bytes()).hexdigest()
            != self.cfg["policy_sha256"]
        ):
            raise ValueError("Policy SHA256 does not match metadata")
        names = self.cfg["joint_names"]
        if len(names) != 16 or set(names) != set(JOINTS):
            raise ValueError("Expected all 16 Go2W joints exactly once")
        self.to_canonical = np.array([names.index(name) for name in JOINTS])
        expected_sdk = [int(POLICY_TO_SDK[list(JOINTS).index(n)]) for n in names]
        if self.cfg["policy_to_sdk"] != expected_sdk:
            raise ValueError("Policy joint names disagree with policy_to_sdk")
        self.leg_indices = np.array(
            [i for i, n in enumerate(names) if not n.endswith("_foot_joint")]
        )
        self.wheel_indices = np.array(
            [i for i, n in enumerate(names) if n.endswith("_foot_joint")]
        )
        self.obs_dim = self.cfg["num_one_step_obs"]
        if (
            self.cfg["history_length"],
            self.cfg["num_actions"],
            self.cfg["history_order"],
            self.cfg["history_reset"],
        ) != (6, 16, "newest_first", "zeros_then_first_frame"):
            raise ValueError("Expected six newest-first frames and 16 actions")
        profile = self.cfg.get("observation_profile", "him_go2w_53")
        if (profile, self.obs_dim) not in [
            ("him_go2w_53", 53),
            ("trackinbit_go2w_57", 57),
        ]:
            raise ValueError("Unknown Go2W observation profile")
        self.profile = profile
        if profile == "him_go2w_53" and names != list(JOINTS):
            raise ValueError(
                "The 53-dimensional profile requires canonical joint order"
            )
        if profile == "trackinbit_go2w_57":
            reference_names = [
                f"{leg}_{joint}_joint"
                for leg in ("FL", "FR", "RL", "RR")
                for joint in ("hip", "thigh", "calf", "foot")
            ]
            if names != reference_names:
                raise ValueError(
                    "The reference profile requires interleaved leg/wheel order"
                )
        if not np.allclose(
            np.asarray(self.cfg["torque_limits"])[self.to_canonical], LIMITS
        ):
            raise ValueError(
                "Policy torque limits disagree with the MuJoCo feedback model"
            )
        if self.cfg.get("wheel_velocity_observation", "real") not in (
            "real",
            "training_masked",
        ):
            raise ValueError("Unknown wheel velocity observation mode")
        if abs(self.cfg["policy_dt"] - 0.02) > 1e-9:
            raise ValueError(
                "This simulator currently supports the 50 Hz policy contract"
            )
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS unavailable; use --device cpu only for an explicit CPU comparison"
            )
        self.device = torch.device(device)
        torch.set_num_threads(1)
        self.model = torch.jit.load(str(self.path), map_location=self.device).eval()
        self.history = np.zeros((HISTORY, self.obs_dim), dtype=np.float32)
        self.last_action = np.zeros(16, dtype=np.float32)
        self.latencies = []
        self.nominal = np.asarray(self.cfg["default_leg_angles"], dtype=np.float32)
        self.kp = np.asarray(self.cfg["kp"])[self.to_canonical]
        self.kd = np.asarray(self.cfg["kd"])[self.to_canonical]
        for _ in range(5):
            self.infer(np.zeros((1, HISTORY * self.obs_dim), dtype=np.float32))

    @torch.inference_mode()
    def infer(self, history):
        start = time.perf_counter()
        tensor = torch.from_numpy(history).to(self.device)
        output = self.model(tensor)
        # .cpu() completes MPS execution and includes transfer cost in latency.
        action = output.detach().cpu().numpy()
        if action.shape != (1, 16) or not np.isfinite(action).all():
            raise ValueError("Policy returned nonfinite actions or an incorrect shape")
        return action[0], (time.perf_counter() - start) * 1000

    def step(self, state, command):
        motors = [state.motor_state[int(i)] for i in self.cfg["policy_to_sdk"]]
        q = np.asarray([m.q for m in motors], dtype=np.float32)
        dq = np.asarray([m.dq for m in motors], dtype=np.float32)
        scales = self.cfg
        cmd = np.asarray(command, dtype=np.float32) * scales["command_scale"]
        gyro = np.asarray(state.imu_state.gyroscope) * scales["angular_velocity_scale"]
        gravity = projected_gravity(state.imu_state.quaternion)
        position = np.zeros(16, dtype=np.float32)
        position[self.leg_indices] = (q[self.leg_indices] - self.nominal) * scales[
            "leg_position_scale"
        ]
        velocity = dq * scales["leg_velocity_scale"]
        velocity[self.wheel_indices] = (
            dq[self.wheel_indices] * scales["wheel_velocity_scale"]
        )
        if scales.get("wheel_velocity_observation", "real") == "training_masked":
            velocity[self.wheel_indices] = (
                0  # Observation copy only; never alter simulator state.
            )
        if self.profile == "trackinbit_go2w_57":
            fields = (gyro, gravity, cmd, position, velocity, self.last_action)
        else:
            fields = (
                cmd,
                gyro,
                gravity,
                position[self.leg_indices],
                velocity,
                self.last_action,
            )
        obs = np.concatenate(fields).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite observation")
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = np.clip(
            obs, -scales["clip_observations"], scales["clip_observations"]
        )
        raw, latency = self.infer(self.history.reshape(1, -1))
        self.latencies.append(latency)
        self.last_action = np.clip(raw, -scales["clip_actions"], scales["clip_actions"])
        q_target, dq_target = np.zeros(16), np.zeros(16)
        q_target[self.leg_indices] = (
            self.nominal
            + scales["leg_action_scale"] * self.last_action[self.leg_indices]
        )
        dq_target[self.wheel_indices] = (
            scales["wheel_action_scale"] * self.last_action[self.wheel_indices]
        )
        return q_target[self.to_canonical], dq_target[self.to_canonical]
