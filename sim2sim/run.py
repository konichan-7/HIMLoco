"""Start two separate Python SDK participants: simulator and controller."""

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .config import MODEL, ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument(
        "--model",
        type=Path,
        default=MODEL,
        help="MuJoCo scene (default: resources/robots/go2w/xml/go2w.xml)",
    )
    p.add_argument("--domain", type=int, default=71)
    p.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--schedule", type=Path)
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if (
        not np.isfinite(args.duration)
        or args.duration < 0
        or not 1 <= args.domain <= 200
    ):
        p.error("Use finite duration >= 0 and a nonzero simulation DDS domain <= 200")
    policy = args.policy
    output = args.output or ROOT / "logs/sim2sim" / datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S-%f"
    )
    output.mkdir(parents=True, exist_ok=True)
    output = output.resolve()
    if not args.model.exists():
        p.error("Model missing; run python -m sim2sim.build_model first")
    # Keep the policy and its metadata together for each run.
    inputs = output / "inputs"
    inputs.mkdir(exist_ok=True)
    shutil.copyfile(policy, inputs / "policy.pt")
    shutil.copyfile(
        args.metadata or policy.with_name("metadata.json"), inputs / "metadata.json"
    )
    policy = inputs / "policy.pt"
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CYCLONEDDS_HOME", str(ROOT / ".deps/cyclonedds-install"))
    executable = sys.executable
    if platform.system() == "Darwin" and not args.headless:
        executable = str(Path(sys.executable).parent / "mjpython")
    sim = [
        executable,
        "-m",
        "sim2sim.simulator",
        "--model",
        str(args.model.resolve()),
        "--domain",
        str(args.domain),
        "--duration",
        str(args.duration),
        "--report",
        str(output / "simulator.json"),
    ]
    if args.headless:
        sim.append("--headless")
    control = [
        sys.executable,
        "-m",
        "sim2sim.controller",
        "--policy",
        str(policy.resolve()),
        "--device",
        args.device,
        "--domain",
        str(args.domain),
        "--duration",
        str(args.duration),
        "--report",
        str(output / "controller.json"),
        "--command",
        *map(str, args.command),
    ]
    control += ["--metadata", str(inputs / "metadata.json")]
    if args.schedule:
        shutil.copyfile(args.schedule, inputs / "schedule.json")
        control += ["--schedule", str(inputs / "schedule.json")]
    print(f"Reports: {output}", flush=True)
    processes = []
    simulator_finished_at = None
    try:
        for command in [sim, control]:
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env))
        limit = time.monotonic() + max(60.0, args.duration * 4 + 30)
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in [None, 0] for proc in processes):
                raise RuntimeError("A simulation participant failed")
            if args.duration > 0 and time.monotonic() > limit:
                raise TimeoutError("Local test exceeded its wall-clock budget")
            if processes[0].poll() == 0 and simulator_finished_at is None:
                simulator_finished_at = time.monotonic()
            if (
                simulator_finished_at is not None
                and processes[1].poll() is None
                and time.monotonic() - simulator_finished_at > 1.0
            ):
                # Let a duration-matched controller exit before handling a closed viewer.
                processes[1].send_signal(2)
            time.sleep(0.1)
        if any(proc.returncode != 0 for proc in processes):
            raise RuntimeError("A simulation participant failed")
    except KeyboardInterrupt:
        for proc in processes:
            if proc.poll() is None:
                proc.send_signal(2)
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
    if (output / "controller.json").exists() and (output / "simulator.json").exists():
        summary = {
            "controller": json.loads((output / "controller.json").read_text()),
            "simulator": json.loads((output / "simulator.json").read_text()),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
