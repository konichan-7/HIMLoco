"""TorchScript inference on MPS, plus an explicitly untrained interface probe."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import CACHE, HISTORY, JOINTS, OBS_DIM, POLICY_TO_SDK, metadata


class InterfaceProbe(nn.Module):
    """HIM-shaped MLP for transport/MPS validation; not a learned locomotion policy."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(318, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 19),
        )
        self.actor = nn.Sequential(
            nn.Linear(72, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 16),
        )
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)

    def forward(self, history):
        encoded = self.encoder(history)
        embedding = torch.cat(
            (history[:, :53], encoded[:, :3], F.normalize(encoded[:, 3:], dim=-1)),
            dim=-1,
        )
        actions = self.actor(embedding)
        # Manual low-speed rolling reference; the legs keep nominal targets.
        vx = history[:, 0] / 2.0
        yaw = history[:, 2] / 0.25
        left = (vx - 0.19 * yaw) / (0.086 * 10.0)
        right = (vx + 0.19 * yaw) / (0.086 * 10.0)
        wheels = torch.stack((left, right, left, right), dim=-1)
        return torch.cat((actions[:, :12], wheels), dim=-1)


def make_probe():
    folder = CACHE / "interface_probe"
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1)
    torch.jit.script(InterfaceProbe().eval()).save(str(folder / "policy.pt"))
    (folder / "metadata.json").write_text(
        json.dumps(metadata("interface_probe_untrained"), indent=2) + "\n"
    )
    return folder / "policy.pt"


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
            self.cfg["joint_names"] != list(JOINTS)
            or self.cfg["policy_to_sdk"] != POLICY_TO_SDK.tolist()
        ):
            raise ValueError(
                "Policy/SDK joint order differs from the local Go2W contract"
            )
        if (
            self.cfg["num_one_step_obs"],
            self.cfg["history_length"],
            self.cfg["num_actions"],
            self.cfg["history_order"],
            self.cfg["history_reset"],
        ) != (53, 6, 16, "newest_first", "zeros_then_first_frame"):
            raise ValueError("Expected the proposed Go2W 53 x 6 -> 16 policy contract")
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
        self.history = np.zeros((HISTORY, OBS_DIM), dtype=np.float32)
        self.last_action = np.zeros(16, dtype=np.float32)
        self.latencies = []
        self.nominal = np.asarray(self.cfg["default_leg_angles"], dtype=np.float32)
        for _ in range(5):
            self.infer(np.zeros((1, 318), dtype=np.float32))

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
        motors = [state.motor_state[int(i)] for i in POLICY_TO_SDK]
        q = np.asarray([m.q for m in motors], dtype=np.float32)
        dq = np.asarray([m.dq for m in motors], dtype=np.float32)
        scales = self.cfg
        obs = np.concatenate(
            (
                np.asarray(command, dtype=np.float32) * scales["command_scale"],
                np.asarray(state.imu_state.gyroscope)
                * scales["angular_velocity_scale"],
                projected_gravity(state.imu_state.quaternion),
                (q[:12] - self.nominal) * scales["leg_position_scale"],
                dq[:12] * scales["leg_velocity_scale"],
                dq[12:] * scales["wheel_velocity_scale"],
                self.last_action,
            )
        ).astype(np.float32)
        if not np.isfinite(obs).all():
            raise ValueError("Nonfinite observation")
        self.history[1:] = self.history[:-1].copy()
        self.history[0] = np.clip(
            obs, -scales["clip_observations"], scales["clip_observations"]
        )
        raw, latency = self.infer(self.history.reshape(1, -1))
        self.latencies.append(latency)
        self.last_action = np.clip(raw, -scales["clip_actions"], scales["clip_actions"])
        q_target = np.r_[
            self.nominal + scales["leg_action_scale"] * self.last_action[:12],
            np.zeros(4),
        ]
        dq_target = np.r_[
            np.zeros(12), scales["wheel_action_scale"] * self.last_action[12:]
        ]
        return q_target, dq_target

    def reset(self):
        self.history.fill(0)
        self.last_action.fill(0)
