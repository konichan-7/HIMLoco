"""Real Unitree SDK channels on a simulation-only loopback DDS domain."""

import os
import platform
import threading
import time

import numpy as np

from .config import POLICY_TO_SDK, ROOT

native = ROOT / ".deps/cyclonedds-install"
if native.exists():
    os.environ.setdefault("CYCLONEDDS_HOME", str(native))

from unitree_sdk2py.core import channel
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__LowCmd_,
    unitree_go_msg_dds__LowState_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC

__all__ = [
    "CRC",
    "ChannelPublisher",
    "ChannelSubscriber",
    "Latest",
    "LowCmd_",
    "LowState_",
    "command_arrays",
    "command_message",
    "initialize",
    "unitree_go_msg_dds__LowState_",
]


def initialize(domain=71):
    if not 1 <= domain <= 200:
        raise ValueError("Local simulator requires a nonzero DDS domain in [1, 200].")
    interface = "lo0" if platform.system() == "Darwin" else "lo"
    # Configure the SDK factory before initialization; leave upstream files intact.
    channel.ChannelConfigHasInterface = f"""<CycloneDDS><Domain Id="any">
      <General><Interfaces><NetworkInterface name="{interface}" multicast="false"/></Interfaces>
      <AllowMulticast>false</AllowMulticast></General>
      <Discovery><ParticipantIndex>auto</ParticipantIndex><MaxAutoParticipantIndex>20</MaxAutoParticipantIndex>
      <Peers><Peer Address="127.0.0.1"/></Peers></Discovery>
    </Domain></CycloneDDS>"""
    channel.ChannelFactoryInitialize(domain, interface)


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None
        self.received_at = 0.0
        self.count = 0

    def put(self, value):
        with self.lock:
            self.value, self.received_at = value, time.monotonic()
            self.count += 1

    def get(self):
        with self.lock:
            return self.value, self.received_at, self.count


def command_message(q_target, dq_target, kp, kd):
    msg = unitree_go_msg_dds__LowCmd_()
    msg.head = [0xFE, 0xEF]
    msg.level_flag = 0xFF
    for i, sdk_index in enumerate(POLICY_TO_SDK):
        motor = msg.motor_cmd[int(sdk_index)]
        motor.mode = 1
        motor.q, motor.dq, motor.kp, motor.kd = map(
            float, (q_target[i], dq_target[i], kp[i], kd[i])
        )
        motor.tau = 0.0
    msg.crc = CRC().Crc(msg)
    return msg


def command_arrays(msg):
    if list(msg.head) != [0xFE, 0xEF] or CRC().Crc(msg) != msg.crc:
        raise ValueError("Invalid LowCmd header or CRC")
    motors = [msg.motor_cmd[int(i)] for i in POLICY_TO_SDK]
    values = np.array([[m.q, m.dq, m.kp, m.kd, m.tau] for m in motors], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:, 2:4] < 0):
        raise ValueError("Invalid motor target or gain")
    # Unitree stop sentinels disable their corresponding feedback terms.
    values[np.abs(values[:, 0]) > 1e8, 2] = 0
    values[np.abs(values[:, 1]) > 1e4, 3] = 0
    values[np.array([m.mode != 1 for m in motors])] = 0
    return values
