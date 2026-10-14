import mujoco
import numpy as np
import pytest

from sim2sim.build_model import initial_data
from sim2sim.config import JOINTS, KD, KP, LIMITS, MODEL, NOMINAL, POLICY_TO_SDK
from sim2sim.policy import Policy, make_probe, projected_gravity
from sim2sim.sdk import (
    LowCmd_,
    command_arrays,
    command_message,
    unitree_go_msg_dds__LowState_,
)
from sim2sim.simulator import motor_torques


def test_sdk_mapping_crc_and_reserved_motors():
    targets = np.arange(16) / 10
    msg = command_message(targets, -targets, KP, KD)
    assert msg.motor_cmd[0].q == pytest.approx(targets[3])  # SDK FR hip
    assert msg.motor_cmd[13].dq == pytest.approx(-targets[12])  # SDK FL wheel
    assert all(m.mode == 0 for m in msg.motor_cmd[16:])
    values = command_arrays(msg)
    np.testing.assert_allclose(values[:, 0], targets)
    # Wire deserialization represents octet arrays as bytes, and floats as float32.
    wire_message = LowCmd_.deserialize(msg.serialize())
    np.testing.assert_allclose(command_arrays(wire_message)[:, 0], targets, atol=1e-7)
    msg.motor_cmd[0].q += 1
    with pytest.raises(ValueError, match="CRC"):
        command_arrays(msg)


def test_pd_velocity_mode_and_saturation():
    q = np.r_[NOMINAL, [100.0, -100.0, 3.0, -3.0]]
    dq = np.zeros(16)
    desired_vel = np.r_[np.zeros(12), [2.0, 2.0, 2.0, 2.0]]
    values = command_arrays(
        command_message(np.r_[NOMINAL, np.zeros(4)], desired_vel, KP, KD)
    )
    tau = motor_torques(values, q, dq)
    np.testing.assert_allclose(tau[:12], 0)
    np.testing.assert_allclose(tau[12:], 1.0)  # Wheel angles have no restoring torque.
    values[:, 4] = 1e6
    np.testing.assert_allclose(motor_torques(values, q, dq), LIMITS)


def test_wxyz_gravity_and_invalid_quaternion():
    np.testing.assert_allclose(projected_gravity([1, 0, 0, 0]), [0, 0, -1])
    np.testing.assert_allclose(
        projected_gravity([2**-0.5, 0, 2**-0.5, 0]), [1, 0, 0], atol=1e-6
    )
    with pytest.raises(ValueError):
        projected_gravity([0, 0, 0, 0])


def test_history_timing_and_wheel_observation():
    path = make_probe()
    policy = Policy(path, "cpu")
    state = unitree_go_msg_dds__LowState_()
    state.imu_state.quaternion = [1.0, 0.0, 0.0, 0.0]
    for i, sdk_index in enumerate(POLICY_TO_SDK):
        state.motor_state[int(sdk_index)].q = float(NOMINAL[i]) if i < 12 else 1000.0
        state.motor_state[int(sdk_index)].dq = float(i)
    policy.step(state, [0.2, 0, 0])
    previous = policy.last_action.copy()
    first_frame = policy.history[0].copy()
    assert np.all(policy.history[1:] == 0)
    np.testing.assert_allclose(first_frame[9:21], 0)
    np.testing.assert_allclose(first_frame[33:37], np.arange(12, 16) * 0.05)
    policy.step(state, [0.1, 0, 0])
    np.testing.assert_allclose(policy.history[0, 37:53], previous)
    np.testing.assert_allclose(policy.history[1], first_frame)
    policy.reset()
    assert np.all(policy.history == 0) and np.all(policy.last_action == 0)


def test_model_preserves_mass_and_joint_axes():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    assert (model.nq, model.nv, model.nu) == (23, 22, 16)
    assert model.body_mass.sum() == pytest.approx(19.126408, abs=1e-6)
    assert {model.joint(name).id for name in JOINTS} == set(range(1, 17))
    for name in JOINTS[12:]:
        joint = model.joint(name)
        assert not joint.limited[0]
        np.testing.assert_allclose(joint.axis, [0, 1, 0])
    data = initial_data(model)
    assert data.qpos[2] == pytest.approx(0.6)
    assert np.isfinite(data.qpos).all()
