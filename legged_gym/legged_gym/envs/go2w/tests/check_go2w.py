"""Tensor regressions for Gym callbacks; no PhysX runtime is simulated here."""
import ast
import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import unittest
import torch

ROOT = Path(__file__).resolve().parents[4]


def inverse_rotate(q, v):
    """Inverse rotation for unit xyzw quaternions (Gym's convention)."""
    xyz, w = q[:, :3], q[:, 3:4]
    return v * (2 * w.square() - 1) - 2 * w * torch.cross(xyz, v, dim=1) + 2 * xyz * (xyz * v).sum(1, keepdim=True)


def load_definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(tree, str(ROOT / path), 'exec'), namespace)


def make_env():
    ns = {
        'torch': torch, 'inspect': inspect, 'BaseTask': object,
        'quat_rotate_inverse': inverse_rotate,
        'torch_rand_float': lambda lo, hi, shape, device: lo + (hi - lo) * torch.rand(shape, device=device),
        'gymtorch': NS(unwrap_tensor=lambda x: x),
        'gymapi': NS(Vec3=lambda x, y, z: NS(x=x, y=y, z=z)),
    }
    load_definitions('legged_gym/envs/base/base_config.py', ['BaseConfig'], ns)
    load_definitions('legged_gym/envs/base/legged_robot_config.py', ['LeggedRobotCfg', 'LeggedRobotCfgPPO'], ns)
    load_definitions('legged_gym/envs/go2w/go2w_config.py', ['Go2wRoughCfg'], ns)
    load_definitions('legged_gym/envs/base/legged_robot.py', ['LeggedRobot'], ns)
    load_definitions('legged_gym/envs/go2w/go2w_robot.py', ['Go2w'], ns)
    robot = ns['Go2w'].__new__(ns['Go2w'])
    robot.cfg = ns['Go2wRoughCfg']()
    # Isolate tensor-reset checks from terrain/DR sampling. Default settings
    # are checked independently below.
    robot.cfg.terrain.curriculum = False
    robot.cfg.commands.curriculum = False
    robot.cfg.domain_rand.randomize_initial_joint_pos = False
    robot.cfg.domain_rand.initial_base_velocity_range = [0., 0.]
    robot.Kp_factors = robot.Kd_factors = robot.motor_strength_factors = torch.ones(2, 1)
    robot._test_namespace = ns
    robot.num_envs, robot.num_bodies = 2, 1
    robot.num_actions, robot.num_one_step_obs = 16, 53
    robot.num_privileged_obs, robot.num_obs = 246, 318
    robot.device, robot.sim, robot.dt = 'cpu', None, .02
    robot.obs_scales = robot.cfg.normalization.obs_scales
    robot.root_states = torch.zeros(2, 13)
    robot.root_states[:, 6] = 1
    robot.base_quat = robot.root_states[:, 3:7]
    robot.base_init_state = torch.tensor([0., 0., .42, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
    robot.base_lin_vel = torch.ones(2, 3)
    robot.base_ang_vel = torch.ones(2, 3) * 2
    robot.projected_gravity = torch.ones(2, 3)
    robot.gravity_vec = torch.tensor([[0., 0., -1.]] * 2)
    robot.env_origins = torch.zeros(2, 3)
    robot.custom_origins = False
    robot.policy_dof_indices = torch.tensor([7, 0, 13, 3, 10, 5, 15, 2, 9, 6, 12, 1, 14, 4, 11, 8])
    robot.leg_dof_indices = robot.policy_dof_indices[:12]
    robot.wheel_dof_indices = robot.policy_dof_indices[12:]
    robot.default_dof_pos = torch.zeros(1, 16)
    robot.default_dof_pos[:, robot.leg_dof_indices] = torch.tensor([0., .8, -1.5] * 4)
    robot.dof_state = torch.ones(2, 16, 2)
    robot.dof_pos, robot.dof_vel = robot.dof_state[..., 0], robot.dof_state[..., 1]
    robot.actions = torch.ones(2, 16)
    robot.last_actions = torch.ones(2, 16)
    robot.last_last_actions = torch.ones(2, 16)
    robot.last_dof_vel = torch.ones(2, 16)
    robot.last_root_vel = torch.zeros(2, 6)
    robot.obs_buf = torch.ones(2, 318)
    robot.commands = torch.zeros(2, 3)
    robot.commands_scale = torch.tensor([2., 2., .25])
    robot.disturbance = torch.zeros(2, 1, 3)
    robot.measured_heights = torch.zeros(2, 187)
    robot.noise_scale_vec = robot._get_noise_scale_vec(robot.cfg)
    robot.add_noise = False
    robot.feet_indices = torch.tensor([0])
    robot.feet_air_time = torch.zeros(2, 1)
    robot.rigid_body_states = torch.zeros(2, 1, 13)
    robot.reset_buf = torch.zeros(2, dtype=torch.bool)
    robot.episode_length_buf = torch.ones(2, dtype=torch.long)
    robot.time_out_buf = torch.zeros(2, dtype=torch.bool)
    robot.common_step_counter = 0
    robot.extras, robot.episode_sums = {}, {}
    robot.viewer = None
    robot.gym = NS(**{name: lambda *args: None for name in (
        'set_dof_state_tensor_indexed', 'set_actor_root_state_tensor_indexed',
        'set_actor_root_state_tensor', 'refresh_actor_root_state_tensor',
        'refresh_net_contact_force_tensor', 'refresh_rigid_body_state_tensor',
    )})
    robot._get_heights = lambda: torch.zeros(2, 187)
    robot._resample_commands = lambda ids: None
    robot.refresh_actor_rigid_shape_props = lambda ids: None
    return robot


def test_reset_refreshes_first_frame_and_preserves_other_env(env, randomized):
    if randomized:
        env.cfg = env._test_namespace['Go2wRoughCfg']()
        env.cfg.terrain.curriculum = False
        env.cfg.commands.curriculum = False
        env.Kp_factors = env.Kd_factors = env.motor_strength_factors = torch.ones(2, 1)
    other = [value[1].clone() for value in (env.obs_buf, env.root_states, env.base_lin_vel, env.base_ang_vel, env.projected_gravity)]
    env.reset_idx(torch.tensor([0]))
    for value, expected in zip((env.obs_buf, env.root_states, env.base_lin_vel, env.base_ang_vel, env.projected_gravity), other):
        torch.testing.assert_close(value[1], expected)
    env.compute_observations()
    torch.testing.assert_close(env.obs_buf[0, 3:6], env.root_states[0, 10:13] * .25)
    torch.testing.assert_close(env.obs_buf[0, 6:9], torch.tensor([0., 0., -1.]))
    torch.testing.assert_close(env.privileged_obs_buf[0, 53:56], env.root_states[0, 7:10] * 2)
    assert torch.count_nonzero(env.obs_buf[0, 53:]) == 0
    assert torch.count_nonzero(env.obs_buf[0, 37:53]) == 0
    env.reset_idx(torch.empty(0, dtype=torch.long))


def test_terminal_observation_survives_reset(env):
    env.root_states[0, 3:7] = torch.tensor([2 ** -.5, 0., 0., 2 ** -.5])
    env.root_states[0, 7:13] = torch.tensor([1., 2., 3., 4., 5., 6.])
    old_root = env.root_states.clone()
    env._post_physics_step_callback = lambda: None
    env.check_termination = lambda: env.reset_buf.copy_(torch.tensor([True, False]))
    env.compute_reward = lambda: None
    ids, terminal = env.post_physics_step()
    assert ids.tolist() == [0]
    torch.testing.assert_close(terminal[:, 53:56], inverse_rotate(old_root[:1, 3:7], old_root[:1, 7:10]) * 2)
    torch.testing.assert_close(terminal[:, 3:6], inverse_rotate(old_root[:1, 3:7], old_root[:1, 10:13]) * .25)
    torch.testing.assert_close(terminal[:, 6:9], torch.tensor([[0., -1., 0.]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(env.obs_buf[0, 6:9], torch.tensor([0., 0., -1.]))
    assert torch.count_nonzero(env.privileged_obs_buf[0, 53:56]) == 0
    assert torch.count_nonzero(env.obs_buf[0, 53:]) == 0
    assert torch.all(env.obs_buf[1, 53:] == 1)


def test_push_updates_rotated_velocity_before_reward_and_him_target(env):
    env.cfg.domain_rand.max_push_vel_xy = .5
    env.root_states[:, 3:7] = torch.tensor([0., 0., 2 ** -.5, 2 ** -.5])
    env.root_states[:, 9] = .2
    env._test_namespace['torch_rand_float'] = lambda *args, **kwargs: torch.tensor([[.3, -.4], [.3, -.4]])
    env._push_robots()
    expected = torch.tensor([[-.4, -.3, .2]] * 2)
    torch.testing.assert_close(env.base_lin_vel, expected)
    frame = env._current_actor_observation()
    torch.testing.assert_close(env._critic_observation(frame)[:, 53:56], expected * 2)
    env.commands[:, :2] = expected[:, :2]
    torch.testing.assert_close(env._reward_tracking_lin_vel(), torch.ones(2))


def test_com_offset_uses_loaded_nominal_without_accumulation(env, displacement):
    env.cfg.domain_rand.randomize_com_displacement = True
    env.cfg.domain_rand.randomize_payload_mass = False
    # Deliberately differs from the URDF: Gym can merge fixed links.
    env.default_base_com = (.031, -.007, -.012)
    env.com_displacement = torch.tensor([displacement, displacement])
    for i in range(2):
        props = [NS(com=NS(x=.031, y=-.007, z=-.012), mass=10.)]
        env._process_rigid_body_props(props, i)
        expected = torch.tensor(env.default_base_com) + env.com_displacement[i]
        torch.testing.assert_close(torch.tensor([props[0].com.x, props[0].com.y, props[0].com.z]), expected)


def test_com_unchanged_when_randomization_disabled(env):
    env.cfg.domain_rand.randomize_payload_mass = False
    env.cfg.domain_rand.randomize_com_displacement = False
    props = [NS(com=NS(x=.031, y=-.007, z=-.012), mass=10.)]
    env._process_rigid_body_props(props, 0)
    assert (props[0].com.x, props[0].com.y, props[0].com.z) == (.031, -.007, -.012)


def test_original_him_files_are_unchanged():
    import subprocess
    repo = ROOT.parent
    for name in [
        'legged_gym/legged_gym/envs/base/legged_robot.py',
        'legged_gym/legged_gym/scripts/train.py',
        'legged_gym/legged_gym/scripts/play.py',
        'rsl_rl/rsl_rl/modules/him_estimator.py',
        'rsl_rl/rsl_rl/modules/him_actor_critic.py',
        'rsl_rl/rsl_rl/runners/him_on_policy_runner.py',
    ]:
        assert (repo / name).read_bytes() == subprocess.check_output(['git', 'show', 'HEAD:' + name], cwd=repo)


def test_single_task_inherits_original_terrain_commands_and_dr(env):
    ns = env._test_namespace
    cfg, original = ns['Go2wRoughCfg'](), ns['LeggedRobotCfg']()
    for section in ('terrain', 'commands', 'domain_rand'):
        baseline = getattr(original, section)
        actual = getattr(cfg, section)
        for name in dir(baseline):
            if name.startswith('_'):
                continue
            expected, value = getattr(baseline, name), getattr(actual, name)
            if hasattr(expected, '__dict__'):
                for field in dir(expected):
                    if not field.startswith('_'):
                        assert getattr(value, field) == getattr(expected, field)
            else:
                assert value == expected
    assert cfg.terrain.mesh_type == 'trimesh' and cfg.terrain.curriculum
    assert cfg.domain_rand.randomize_friction and cfg.domain_rand.delay
    registry = (ROOT / 'legged_gym/envs/__init__.py').read_text()
    assert 'register("go2w", Go2w, Go2wRoughCfg(), Go2wRoughCfgPPO())' in registry
    assert 'go2w_flat' not in registry



def test_exact_wheel_contacts_after_fixed_link_merging(env):
    base = env._test_namespace['LeggedRobot']
    base._create_envs = lambda self: None
    env.envs, env.actor_handles = [object()], [0]
    handles = {'FL_foot': 4, 'FR_foot': 8, 'RL_foot': 12, 'RR_foot': 16, 'FL_foot_motor': 3}
    env.gym.find_actor_rigid_body_handle = lambda _, __, name: handles.get(name, -1)
    env._create_envs()
    assert env.feet_indices.tolist() == [4, 8, 12, 16]
    del handles['RR_foot']
    try:
        env._create_envs()
    except ValueError:
        pass
    else:
        raise AssertionError('Missing wheel contact body must fail')


def test_rough_terrain_height_samples_reach_critic(env):
    env.num_height_points = 187
    env.height_points = torch.zeros(2, 187, 3)
    env.height_samples = torch.full((5, 5), 7.)
    env.terrain = NS(cfg=NS(border_size=0., horizontal_scale=1., vertical_scale=.1))
    env._test_namespace['quat_apply_yaw'] = lambda _, points: points
    env.root_states[:, :3] = torch.tensor([1., 1., 1.2])
    # Call the unchanged inherited sampler rather than the reset fixture stub.
    env.measured_heights = env._test_namespace['LeggedRobot']._get_heights(env)
    torch.testing.assert_close(env.measured_heights, torch.full((2, 187), .7))
    critic = env._critic_observation(env._current_actor_observation())
    torch.testing.assert_close(critic[:, 59:], torch.zeros(2, 187), atol=1e-6, rtol=0)


def test_original_him_update_export_and_mapped_control(env):
    import contextlib
    import copy
    import hashlib
    import importlib.util
    import io
    import json
    import sys
    import tempfile
    import xml.etree.ElementTree as ET
    import numpy as np
    import torch.nn.functional as F

    repo = ROOT.parent
    sys.path[:0] = [str(repo), str(repo / 'rsl_rl')]
    from rsl_rl.modules import HIMActorCritic
    from rsl_rl.algorithms import HIMPPO
    from sim2sim.simulator import motor_torques
    ns = env._test_namespace
    ns.update(Path=Path, hashlib=hashlib, ET=ET, LEGGED_GYM_ROOT_DIR=str(ROOT))
    load_definitions('legged_gym/envs/go2w/go2w_config.py', ['go2w_policy_metadata'], ns)
    metadata = ns['go2w_policy_metadata'](env.cfg)
    q = env.dof_pos[:, env.policy_dof_indices].numpy()
    dq = env.dof_vel[:, env.policy_dof_indices].numpy()
    env.torques = torch.zeros(2, 16)
    env.Kp_factors = env.Kd_factors = env.motor_strength_factors = torch.ones(2, 1)
    env.torque_limits = torch.zeros(16)
    env.torque_limits[env.policy_dof_indices] = torch.tensor(metadata['torque_limits'])
    env.actions[:] = .4
    a = env.actions.numpy()
    targets = np.c_[np.asarray(metadata['default_leg_angles']) + .25 * a[:, :12], np.zeros((2, 4))]
    velocities = np.c_[np.zeros((2, 12)), 10 * a[:, 12:]]
    values = np.stack([targets, velocities, np.broadcast_to(metadata['kp'], (2, 16)), np.broadcast_to(metadata['kd'], (2, 16)), np.zeros((2, 16))], axis=2)
    expected = np.stack([motor_torques(values[i], q[i], dq[i]) for i in range(2)])
    np.testing.assert_allclose(env._compute_torques(env.actions)[:, env.policy_dof_indices].numpy(), expected, atol=1e-5)

    spec = importlib.util.spec_from_file_location('go2w_test_helpers', ROOT / 'legged_gym/utils/helpers.py')
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    tree = ast.parse(Path(spec.origin).read_text())
    tree.body = [node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module in ('isaacgym', 'legged_gym'))]
    exec(compile(tree, spec.origin, 'exec'), helper.__dict__)
    ns.update(torch=torch, np=np, json=json, export_policy_as_jit=helper.export_policy_as_jit)
    load_definitions('legged_gym/envs/go2w/export.py', ['export_go2w_policy'], ns)
    torch.manual_seed(7)
    for d, actions, privileged in [(53, 16, 246), (45, 12, 238)]:
        with contextlib.redirect_stdout(io.StringIO()):
            model = HIMActorCritic(d * 6, privileged, d, actions)
        x, successor = torch.randn(8, d * 6), torch.randn(8, privileged)
        captured = []
        hook = model.estimator.target.register_forward_pre_hook(lambda _, args: captured.append(args[0].clone()))
        with torch.no_grad():
            expected_loss = F.mse_loss(model.estimator.encoder(x)[:, :3], successor[:, d:d + 3])
        losses = model.estimator.update(x, successor)
        hook.remove()
        torch.testing.assert_close(captured[0], successor[:, 3:d + 3])
        assert abs(losses[0] - expected_loss.item()) < 1e-6
        with tempfile.TemporaryDirectory() as output:
            if d == 53:
                path = ns['export_go2w_policy'](model, env.cfg, output, x)
                meta = json.loads(path.with_name('metadata.json').read_text())
                assert meta['num_one_step_obs'] == 53 and meta['parity_max_abs_error'] <= 1e-5
                assert meta['policy_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
            else:
                helper.export_policy_as_jit(model, output)
                path = Path(output) / 'policy.pt'
            with torch.no_grad():
                torch.testing.assert_close(torch.jit.load(str(path))(x), model.act_inference(x), atol=1e-6, rtol=0)
        if d == 53:
            ppo = HIMPPO(model, num_learning_epochs=2, num_mini_batches=2)
            ppo.init_storage(8, 4, [318], [246], [16])
            with torch.no_grad():
                for _ in range(4):
                    ppo.act(torch.randn(8, 318), torch.randn(8, 246))
                    ppo.process_env_step(torch.randn(8), torch.zeros(8), {}, torch.randn(8, 246))
                ppo.compute_returns(torch.randn(8, 246))
            assert np.isfinite(ppo.update()).all()


def load_tests(loader, tests, pattern):
    cases = [
        (test_reset_refreshes_first_frame_and_preserves_other_env, {'randomized': False}),
        (test_reset_refreshes_first_frame_and_preserves_other_env, {'randomized': True}),
        (test_terminal_observation_survives_reset, {}),
        (test_push_updates_rotated_velocity_before_reward_and_him_target, {}),
        (test_com_offset_uses_loaded_nominal_without_accumulation, {'displacement': [0., 0., 0.]}),
        (test_com_offset_uses_loaded_nominal_without_accumulation, {'displacement': [.01, -.02, .015]}),
        (test_com_unchanged_when_randomization_disabled, {}),
        (test_single_task_inherits_original_terrain_commands_and_dr, {}),
        (test_exact_wheel_contacts_after_fixed_link_merging, {}),
        (test_rough_terrain_height_samples_reach_critic, {}),
        (test_original_him_update_export_and_mapped_control, {}),
    ]
    suite = unittest.TestSuite()
    for function, kwargs in cases:
        suite.addTest(unittest.FunctionTestCase(
            lambda f=function, kw=kwargs: f(make_env(), **kw),
            description=function.__name__ + str(kwargs),
        ))
    suite.addTest(unittest.FunctionTestCase(test_original_him_files_are_unchanged))
    return suite


if __name__ == '__main__':
    unittest.main(verbosity=2)
