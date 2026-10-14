# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float

from legged_gym.envs.base.legged_robot import LeggedRobot


class Go2w(LeggedRobot):
    """HIM Go2W environment with policy-order actions and explicit wheel control."""

    LEG_JOINT_NAMES = [
        f"{leg}_{joint}_joint"
        for leg in ("FL", "FR", "RL", "RR")
        for joint in ("hip", "thigh", "calf")
    ]
    WHEEL_JOINT_NAMES = [f"{leg}_foot_joint" for leg in ("FL", "FR", "RL", "RR")]
    POLICY_JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES

    def _create_envs(self):
        super()._create_envs()
        # The URDF also contains fixed *_foot_motor bodies. Select only the
        # four rotating wheel bodies, after Gym's fixed-link merging.
        handles = [
            self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], name)
            for name in self.cfg.asset.feet_names
        ]
        if len(handles) != 4 or any(handle < 0 for handle in handles) or len(set(handles)) != 4:
            raise ValueError(f"Go2W wheel contact bodies are missing or duplicated: {handles}")
        self.feet_indices = torch.tensor(handles, dtype=torch.long, device=self.device)

    def _process_rigid_body_props(self, props, env_id):
        if env_id == 0:
            com = props[0].com
            self.default_base_com = (com.x, com.y, com.z)
        props = super()._process_rigid_body_props(props, env_id)
        if self.cfg.domain_rand.randomize_com_displacement:
            x, y, z = self.default_base_com
            dx, dy, dz = self.com_displacement[env_id].tolist()
            props[0].com = gymapi.Vec3(x + dx, y + dy, z + dz)
        return props

    def _refresh_base_state(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        self.base_quat[indices] = self.root_states[indices, 3:7]
        quat = self.base_quat[indices]
        self.base_lin_vel[indices] = quat_rotate_inverse(quat, self.root_states[indices, 7:10])
        self.base_ang_vel[indices] = quat_rotate_inverse(quat, self.root_states[indices, 10:13])
        self.projected_gravity[indices] = quat_rotate_inverse(quat, self.gravity_vec[indices])

    def _push_robots(self):
        super()._push_robots()
        self._refresh_base_state()

    def _init_buffers(self):
        super()._init_buffers()
        if self.num_dofs != 16 or self.num_actions != 16:
            raise ValueError(
                f"Go2W expects 16 active DOFs and actions, got {self.num_dofs} DOFs "
                f"and {self.num_actions} actions"
            )
        missing = [name for name in self.POLICY_JOINT_NAMES if name not in self.dof_names]
        if missing:
            raise ValueError(f"Go2W asset is missing policy joints: {missing}")
        self.policy_dof_indices = torch.tensor(
            [self.dof_names.index(name) for name in self.POLICY_JOINT_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.leg_dof_indices = self.policy_dof_indices[:12]
        self.wheel_dof_indices = self.policy_dof_indices[12:]
        if len(self.feet_indices) != 4:
            raise ValueError(
                "Go2W requires exactly FL_foot, FR_foot, RL_foot and RR_foot "
                f"as wheel contact bodies; found {len(self.feet_indices)}"
            )
        if self.num_one_step_obs != 53 or self.num_obs != 318 or self.num_privileged_obs != 246:
            raise ValueError(
                "Go2W interface must be 53 actor observations, 318 history values, "
                "and 246 critic observations; got "
                f"{self.num_one_step_obs}/{self.num_obs}/{self.num_privileged_obs}"
            )

    def _get_noise_scale_vec(self, cfg):
        """Noise scales follow [cmd, gyro, gravity, leg qerr, all qd, action]."""
        noise = torch.zeros(self.num_one_step_obs + 187, device=self.device)
        scales = cfg.noise.noise_scales
        level = cfg.noise.noise_level
        obs = cfg.normalization.obs_scales
        noise[3:6] = scales.ang_vel * level * obs.ang_vel
        noise[6:9] = scales.gravity * level
        noise[9:21] = scales.dof_pos * level * obs.dof_pos
        noise[21:33] = scales.dof_vel * level * obs.dof_vel
        noise[33:37] = getattr(scales, "wheel_vel", scales.dof_vel) * level * obs.dof_vel
        noise[53:] = scales.height_measurements * level * obs.height_measurements
        self.add_noise = cfg.noise.add_noise
        return noise

    def _process_dof_props(self, props, env_id):
        props = super()._process_dof_props(props, env_id)
        if env_id == 0:
            # Continuous wheel joints have no meaningful positional limits.
            for index, name in enumerate(self.dof_names):
                if name in self.WHEEL_JOINT_NAMES:
                    self.dof_pos_limits[index, 0] = -float("inf")
                    self.dof_pos_limits[index, 1] = float("inf")
        return props

    def _policy_order(self, values):
        return values.index_select(1, self.policy_dof_indices)

    def _compute_torques(self, actions):
        """Apply position PD to legs and velocity damping control to wheels."""
        actions = torch.clamp(actions, -self.cfg.normalization.clip_actions, self.cfg.normalization.clip_actions)
        leg_pos = self.dof_pos.index_select(1, self.leg_dof_indices)
        leg_vel = self.dof_vel.index_select(1, self.leg_dof_indices)
        default_leg_pos = self.default_dof_pos.index_select(1, self.leg_dof_indices)

        leg_targets = default_leg_pos + actions[:, :12] * self.cfg.control.leg_action_scale
        leg_torques = (
            self.cfg.control.stiffness["joint"]
            * self.Kp_factors
            * (leg_targets - leg_pos)
            - self.cfg.control.damping["joint"] * self.Kd_factors * leg_vel
        )

        wheel_vel = self.dof_vel.index_select(1, self.wheel_dof_indices)
        wheel_targets = actions[:, 12:] * self.cfg.control.wheel_action_scale
        wheel_targets = torch.clamp(
            wheel_targets,
            -self.cfg.control.wheel_speed_limit,
            self.cfg.control.wheel_speed_limit,
        )
        wheel_torques = (
            self.cfg.control.wheel_velocity_gain
            * self.Kd_factors
            * (wheel_targets - wheel_vel)
        )

        policy_torques = torch.cat((leg_torques, wheel_torques), dim=1)
        sim_torques = torch.zeros_like(self.torques)
        sim_torques[:, self.policy_dof_indices] = policy_torques
        sim_torques *= self.motor_strength_factors
        return torch.maximum(torch.minimum(sim_torques, self.torque_limits), -self.torque_limits)

    def _current_actor_observation(self, add_noise=True):
        leg_pos = self.dof_pos.index_select(1, self.leg_dof_indices)
        joint_vel = self._policy_order(self.dof_vel)
        frame = torch.cat(
            (
                self.commands[:, :3] * self.commands_scale,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                (leg_pos - self.default_dof_pos.index_select(1, self.leg_dof_indices))
                * self.obs_scales.dof_pos,
                joint_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        if frame.shape[1] != self.num_one_step_obs:
            raise RuntimeError(f"Go2W actor observation has {frame.shape[1]} values, expected 53")
        if add_noise and self.add_noise:
            frame = frame + (2.0 * torch.rand_like(frame) - 1.0) * self.noise_scale_vec[:53]
        return frame

    def _critic_observation(self, actor_frame, add_noise=True):
        base_velocity = self.base_lin_vel * self.obs_scales.lin_vel
        external_force = self.disturbance[:, 0, :]
        heights = (
            torch.clamp(self.root_states[:, 2:3] - 0.5 - self.measured_heights, -1.0, 1.0)
            * self.obs_scales.height_measurements
        )
        if add_noise and self.add_noise:
            heights = heights + (2.0 * torch.rand_like(heights) - 1.0) * self.noise_scale_vec[53:]
        critic = torch.cat((actor_frame, base_velocity, external_force, heights), dim=-1)
        if critic.shape[1] != self.num_privileged_obs:
            raise RuntimeError(f"Go2W critic observation has {critic.shape[1]} values, expected 246")
        return critic

    def compute_observations(self):
        frame = self._current_actor_observation()
        self.obs_buf = torch.cat((frame, self.obs_buf[:, : -self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = self._critic_observation(frame)

    def get_current_obs(self):
        return self._critic_observation(self._current_actor_observation())

    def compute_termination_observations(self, env_ids):
        if env_ids.numel() == 0:
            return torch.empty((0, self.num_privileged_obs), device=self.device)
        actor_frame = self._current_actor_observation()
        return self._critic_observation(actor_frame)[env_ids]

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if env_ids.numel() == 0:
            return
        self._refresh_base_state(env_ids)
        # A new episode starts from a zeroed action/history state. The next
        # observation update inserts its first frame at the newest slot.
        self.obs_buf[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0

    def _reset_dofs(self, env_ids):
        if env_ids.numel() == 0:
            return
        if self.cfg.domain_rand.randomize_initial_joint_pos:
            lo, hi = self.cfg.domain_rand.initial_joint_pos_range
            multiplier = torch_rand_float(lo, hi, (len(env_ids), 12), device=self.device)
            leg_default = self.default_dof_pos.index_select(1, self.leg_dof_indices)
            self.dof_pos[env_ids[:, None], self.leg_dof_indices] = leg_default * multiplier
        else:
            self.dof_pos[env_ids[:, None], self.leg_dof_indices] = self.default_dof_pos.index_select(
                1, self.leg_dof_indices
            )
        self.dof_pos[env_ids[:, None], self.wheel_dof_indices] = 0.0
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_root_states(self, env_ids):
        if env_ids.numel() == 0:
            return
        super()._reset_root_states(env_ids)
        lo, hi = self.cfg.domain_rand.initial_base_velocity_range
        if lo == hi:
            self.root_states[env_ids, 7:13] = lo
        else:
            self.root_states[env_ids, 7:13] = torch_rand_float(
                lo, hi, (len(env_ids), 6), device=self.device
            )
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reward_dof_vel(self):
        leg_vel = self.dof_vel.index_select(1, self.leg_dof_indices)
        return torch.sum(torch.square(leg_vel), dim=1)

    def _reward_dof_acc(self):
        current = self.dof_vel.index_select(1, self.leg_dof_indices)
        previous = self.last_dof_vel.index_select(1, self.leg_dof_indices)
        return torch.sum(torch.square((current - previous) / self.dt), dim=1)

    def _reward_wheel_acc(self):
        current = self.dof_vel.index_select(1, self.wheel_dof_indices)
        previous = self.last_dof_vel.index_select(1, self.wheel_dof_indices)
        return torch.sum(torch.square((current - previous) / self.dt), dim=1)

    def _reward_dof_pos_limits(self):
        q = self.dof_pos.index_select(1, self.leg_dof_indices)
        limits = self.dof_pos_limits.index_select(0, self.leg_dof_indices)
        lower = -(q - limits[:, 0]).clamp(max=0.0)
        upper = (q - limits[:, 1]).clamp(min=0.0)
        return torch.sum(lower + upper, dim=1)

    def _reward_dof_vel_limits(self):
        qd = self.dof_vel.index_select(1, self.leg_dof_indices)
        limits = self.dof_vel_limits.index_select(0, self.leg_dof_indices)
        excess = (torch.abs(qd) - limits * self.cfg.rewards.soft_dof_vel_limit).clamp(min=0.0, max=1.0)
        return torch.sum(excess, dim=1)

    def _reward_stand_still(self):
        is_standing = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.1)
            & (torch.abs(self.commands[:, 2]) < 0.1)
        )
        leg_error = self.dof_pos.index_select(1, self.leg_dof_indices) - self.default_dof_pos.index_select(
            1, self.leg_dof_indices
        )
        wheel_speed = self.dof_vel.index_select(1, self.wheel_dof_indices)
        return (
            torch.sum(torch.square(leg_error), dim=1)
            + 0.1 * torch.sum(torch.square(wheel_speed), dim=1)
        ) * is_standing

