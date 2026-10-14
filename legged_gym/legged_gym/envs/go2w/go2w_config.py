# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class Go2wRoughCfg(LeggedRobotCfg):
    """Go2W adaptation with the original HIM terrain and domain randomization."""

    class env(LeggedRobotCfg.env):
        num_one_step_observations = 53
        num_observations = 53 * 6
        num_one_step_privileged_obs = 53 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 16
        episode_length_s = 20

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.42]
        default_joint_angles = {
            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FL_foot_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "FR_foot_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RL_thigh_joint": 0.8,
            "RL_calf_joint": -1.5,
            "RL_foot_joint": 0.0,
            "RR_hip_joint": 0.0,
            "RR_thigh_joint": 0.8,
            "RR_calf_joint": -1.5,
            "RR_foot_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 40.0}
        damping = {"joint": 1.0}
        action_scale = 0.25
        leg_action_scale = 0.25
        wheel_action_scale = 10.0
        wheel_speed_limit = 10.0
        wheel_velocity_gain = 0.5
        wheel_radius = 0.086
        decimation = 10

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf"
        name = "go2w"
        foot_name = "foot"
        feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        penalize_contacts_on = ["base", "thigh", "calf"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = True

    class domain_rand(LeggedRobotCfg.domain_rand):
        # All DR switches/ranges inherit the original HIM defaults.
        initial_base_velocity_range = [-0.5, 0.5]

    class rewards(LeggedRobotCfg.rewards):
        only_positive_rewards = False
        tracking_sigma = 0.25
        base_height_target = 0.40
        max_contact_force = 100.0
        soft_dof_pos_limit = 0.95

        class scales(LeggedRobotCfg.rewards.scales):
            termination = -0.0
            tracking_lin_vel = 1.5
            tracking_ang_vel = 0.75
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -0.5
            base_height = -10.0
            torques = -5.0e-4
            dof_acc = -1.0e-7
            dof_vel = -1.0e-7
            wheel_acc = -1.0e-7
            collision = -1.0
            action_rate = -0.01
            stand_still = -0.5
            feet_air_time = 0.0
            feet_stumble = 0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            foot_clearance = 0.0
            joint_power = 0.0
            smoothness = 0.0

    class normalization(LeggedRobotCfg.normalization):
        clip_actions = 1.0

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.0

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.01
            dof_vel = 1.5
            wheel_vel = 1.5
            lin_vel = 0.0
            ang_vel = 0.2
            gravity = 0.05
            height_measurements = 0.1

    class sim(LeggedRobotCfg.sim):
        dt = 0.002
        substeps = 1


class Go2wRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        experiment_name = "rough_go2w"


def go2w_policy_metadata(env_cfg, policy_kind="trained_him"):
    """Build the sidecar contract consumed by the shared MuJoCo/SDK policy core."""
    leg_names = env_cfg.init_state.default_joint_angles
    policy_joint_names = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ] + [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]
    sdk_joint_names = [
        f"{leg}_{joint}_joint"
        for leg in ("FR", "FL", "RR", "RL")
        for joint in ("hip", "thigh", "calf")
    ] + [f"{leg}_foot_joint" for leg in ("FR", "FL", "RR", "RL")]
    policy_to_sdk = [sdk_joint_names.index(name) for name in policy_joint_names]
    urdf_path = Path(env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
    root = ET.parse(urdf_path).getroot()
    limits = {
        joint.attrib["name"]: joint.find("limit")
        for joint in root.findall("joint")
    }
    torque_limits = [float(limits[name].attrib["effort"]) for name in policy_joint_names]
    leg_count = 12
    kp = [float(env_cfg.control.stiffness["joint"])] * leg_count + [0.0] * 4
    kd = [float(env_cfg.control.damping["joint"])] * leg_count + [
        float(env_cfg.control.wheel_velocity_gain)
    ] * 4
    default_leg_angles = [float(leg_names[name]) for name in policy_joint_names[:leg_count]]
    return {
        "schema_version": 1,
        "policy_kind": policy_kind,
        "joint_names": policy_joint_names,
        "policy_to_sdk": policy_to_sdk,
        "observation_fields": [
            "command",
            "body_angular_velocity",
            "projected_gravity",
            "leg_position_error",
            "joint_velocity",
            "previous_action",
        ],
        "num_one_step_obs": 53,
        "history_length": 6,
        "history_order": "newest_first",
        "history_reset": "zeros_then_first_frame",
        "num_actions": 16,
        "default_leg_angles": default_leg_angles,
        "kp": kp,
        "kd": kd,
        "torque_limits": torque_limits,
        "leg_action_scale": float(env_cfg.control.leg_action_scale),
        "wheel_action_scale": float(env_cfg.control.wheel_action_scale),
        "clip_actions": float(env_cfg.normalization.clip_actions),
        "clip_observations": float(env_cfg.normalization.clip_observations),
        "command_scale": [
            float(env_cfg.normalization.obs_scales.lin_vel),
            float(env_cfg.normalization.obs_scales.lin_vel),
            float(env_cfg.normalization.obs_scales.ang_vel),
        ],
        "angular_velocity_scale": float(env_cfg.normalization.obs_scales.ang_vel),
        "leg_position_scale": float(env_cfg.normalization.obs_scales.dof_pos),
        "leg_velocity_scale": float(env_cfg.normalization.obs_scales.dof_vel),
        "wheel_velocity_scale": float(env_cfg.normalization.obs_scales.dof_vel),
        "physics_dt": float(env_cfg.sim.dt),
        "policy_dt": float(env_cfg.sim.dt * env_cfg.control.decimation),
        "sdk_dt": 0.002,
        "wheel_radius": float(env_cfg.control.wheel_radius),
        "wheel_radius_status": "candidate_unverified",
        "asset_urdf": str(urdf_path),
        "asset_urdf_sha256": hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
        "num_critic_obs": int(env_cfg.env.num_privileged_obs),
    }
