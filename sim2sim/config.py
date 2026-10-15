"""The proposed Go2W policy contract, independent of Isaac Gym."""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "legged_gym/resources/robots/go2w/xml/go2w.xml"
LEGS = ("FL", "FR", "RL", "RR")
JOINTS = tuple(
    f"{leg}_{joint}_joint" for leg in LEGS for joint in ("hip", "thigh", "calf")
) + tuple(f"{leg}_foot_joint" for leg in LEGS)
# For each policy joint, its index in Unitree's LowState / LowCmd.
POLICY_TO_SDK = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 13, 12, 15, 14])
NOMINAL = np.tile([0.0, 0.8, -1.5], 4).astype(np.float32)
KP = np.array([40.0] * 12 + [0.0] * 4, dtype=np.float32)
KD = np.array([1.0] * 12 + [0.5] * 4, dtype=np.float32)
LIMITS = np.array([23.7, 23.7, 35.55] * 4 + [23.7] * 4)
PHYSICS_DT = 0.002
POLICY_DT = 0.02
SDK_DT = 0.002
STATE_DT = 0.01
COMMAND_TIMEOUT = 0.2
OBS_DIM = 53
HISTORY = 6


def metadata(kind="trained_him"):
    return {
        "schema_version": 1,
        "policy_kind": kind,
        "joint_names": list(JOINTS),
        "policy_to_sdk": POLICY_TO_SDK.tolist(),
        "observation_fields": [
            "command",
            "body_angular_velocity",
            "projected_gravity",
            "leg_position_error",
            "joint_velocity",
            "previous_action",
        ],
        "num_one_step_obs": OBS_DIM,
        "history_length": HISTORY,
        "history_order": "newest_first",
        "history_reset": "zeros_then_first_frame",
        "num_actions": 16,
        "default_leg_angles": NOMINAL.tolist(),
        "kp": KP.tolist(),
        "kd": KD.tolist(),
        "torque_limits": LIMITS.tolist(),
        "leg_action_scale": 0.25,
        "wheel_action_scale": 10.0,
        "clip_actions": 1.0,
        "clip_observations": 100.0,
        "command_scale": [2.0, 2.0, 0.25],
        "angular_velocity_scale": 0.25,
        "leg_position_scale": 1.0,
        "leg_velocity_scale": 0.05,
        "wheel_velocity_scale": 0.05,
        "physics_dt": PHYSICS_DT,
        "policy_dt": POLICY_DT,
        "wheel_radius": 0.086,
    }
