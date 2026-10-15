"""Export the pinned TrackinBIT 57/342 checkpoint without Isaac Gym."""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .config import JOINTS, POLICY_TO_SDK, ROOT, metadata

REFERENCE_COMMIT = "011693738c61603c3f22f2bce755098dd36fa7eb"


class InferenceHIM(nn.Module):
    """Same source encoder, normalized latent and actor as act_inference."""

    def __init__(self, model):
        super().__init__()
        self.encoder = copy.deepcopy(model.estimator.encoder)
        self.actor = copy.deepcopy(model.actor)
        self.obs_dim = model.num_one_step_obs

    def forward(self, history):
        encoded = self.encoder(history)
        return self.actor(
            torch.cat(
                (
                    history[:, : self.obs_dim],
                    encoded[:, :3],
                    F.normalize(encoded[:, 3:], dim=-1),
                ),
                dim=-1,
            )
        )


def trackinbit_metadata(wheel_observation="training_masked"):
    cfg = metadata()
    names = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf", "foot")
    ]
    order = [list(JOINTS).index(n) for n in names]
    cfg.update(
        observation_profile="trackinbit_go2w_57",
        num_one_step_obs=57,
        joint_names=names,
        policy_to_sdk=POLICY_TO_SDK[order].tolist(),
        observation_fields=[
            "body_angular_velocity",
            "projected_gravity",
            "command",
            "joint_position_error_wheels_zero",
            "joint_velocity",
            "previous_action",
        ],
        kp=[cfg["kp"][i] for i in order],
        kd=[cfg["kd"][i] for i in order],
        torque_limits=[cfg["torque_limits"][i] for i in order],
        clip_actions=100.0,
        num_critic_obs=262,
        wheel_velocity_observation=wheel_observation,
        source_repository="https://github.com/TrackinBIT/HIMLoco-for-Go2W",
        source_commit=REFERENCE_COMMIT,
        training_physics_dt=0.005,
        physics_dt=0.002,
        pd_dt=0.002,
        policy_dt=0.02,
        transfer_note="Reference trained at 200 Hz PD; requested replay uses 500 Hz PD. "
        "training_masked reproduces reward-induced zero wheel velocity observations only.",
    )
    return cfg


def load_reference(checkpoint):
    sys.path.insert(0, str(ROOT / "rsl_rl"))
    from rsl_rl.modules import HIMActorCritic

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload["model_state_dict"]
    # Strict loading checks all layers, including critic/target/prototype.
    model = HIMActorCritic(342, 262, 57, 16).cpu().eval()
    model.load_state_dict(state, strict=True)
    return model, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--wheel-observation",
        choices=["real", "training_masked"],
        default="training_masked",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    model, payload = load_reference(args.checkpoint)
    args.output.mkdir(parents=True, exist_ok=True)
    policy = torch.jit.script(InferenceHIM(model).eval())
    path = args.output / "policy.pt"
    policy.save(str(path))
    policy = torch.jit.load(str(path)).eval()
    observations = torch.randn(8, 342, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected = model.act_inference(observations)
        actual = policy(observations)
    error = float((expected - actual).abs().max())
    if not torch.isfinite(actual).all() or error > 1e-5:
        raise ValueError(f"HIM export parity failed: {error}")
    cfg = trackinbit_metadata(args.wheel_observation)
    cfg.update(
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        checkpoint_iteration_field=payload.get("iter"),
        policy_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        parity_sample_count=len(observations),
        parity_max_abs_error=error,
    )
    (args.output / "metadata.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"Exported {path}; max parity error={error:g}")


if __name__ == "__main__":
    main()
