"""Native runtime, real SDK, and CPU/MPS numerical/latency checks."""

import importlib.metadata
import json
import platform
import subprocess
import time

import numpy as np
import torch

from .config import CACHE, MODEL, ROOT
from .policy import InterfaceProbe
from .sdk import CRC, unitree_go_msg_dds__LowCmd_


@torch.inference_mode()
def benchmark():
    torch.set_num_threads(1)
    torch.manual_seed(2)
    network = InterfaceProbe().eval()
    # Nonzero leg outputs exercise the complete encoder/actor computation.
    torch.nn.init.normal_(network.actor[-1].weight, std=0.01)
    inputs = torch.randn(1, 318)
    reference = network(inputs).numpy()
    timing = {}
    errors = {}
    for device in ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"]:
        model = torch.jit.script(network).to(device)
        sample = inputs.to(device)
        for _ in range(10):
            model(sample).cpu()
        elapsed = []
        for _ in range(100):
            start = time.perf_counter()
            # Include host/device transfer and synchronization, as deployment does.
            result = model(inputs.to(device)).cpu().numpy()
            elapsed.append((time.perf_counter() - start) * 1000)
        errors[device] = float(np.max(np.abs(result - reference)))
        if errors[device] > 1e-4:
            raise RuntimeError(f"{device} output differs from CPU by {errors[device]}")
        timing[device] = {
            str(p): float(np.percentile(elapsed, p)) for p in [50, 95, 99, 100]
        }
    return {
        "batch_size": 1,
        "network": "untrained HIM-shaped numerical probe",
        "max_abs_error": errors,
        "latency_ms": timing,
    }


def main():
    versions = {
        name: importlib.metadata.version(name)
        for name in ["torch", "mujoco", "numpy", "cyclonedds", "unitree-sdk2py"]
    }
    sdk_sha = subprocess.check_output(
        ["git", "-C", str(ROOT / ".deps/unitree_sdk2_python"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    report = {
        "platform": platform.system(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "versions": versions,
        "sdk_commit": sdk_sha,
        "cyclonedds_native": "0.10.5",
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "sdk_crc_default_lowcmd": CRC().Crc(unitree_go_msg_dds__LowCmd_()),
        "model_exists": MODEL.exists(),
        "inference_check": benchmark(),
    }
    if (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and not report["mps_available"]
    ):
        raise RuntimeError("Apple Silicon MPS acceleration is not available")
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "environment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
