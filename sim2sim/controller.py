"""Python deployment core connected only to the local simulated Go2W."""

import argparse
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np

from .config import POLICY_DT, SDK_DT
from .policy import Policy
from .sdk import (
    ChannelPublisher,
    ChannelSubscriber,
    Latest,
    LowCmd_,
    LowState_,
    command_message,
    initialize,
)


def run(args):
    policy = Policy(args.policy, args.device, args.metadata)
    print(
        f"POLICY kind={policy.cfg['policy_kind']} device={policy.device} path={policy.path}",
        flush=True,
    )
    initialize(args.domain)
    state = Latest()
    subscriber = ChannelSubscriber("rt/lowstate", LowState_)
    subscriber.Init(state.put, 1)
    publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
    publisher.Init()
    outgoing = Latest()
    stopped = threading.Event()
    writer_stats = {"writes": 0, "overruns": 0, "error": None}

    def publish_targets():
        deadline = time.monotonic()
        while not stopped.is_set():
            target, updated_at, _ = outgoing.get()
            if target is not None and time.monotonic() - updated_at <= 0.2:
                if not publisher.Write(target):
                    writer_stats["error"] = "LowCmd publication failed"
                    break
                writer_stats["writes"] += 1
            deadline += SDK_DT
            delay = deadline - time.monotonic()
            if delay > 0:
                stopped.wait(delay)
            else:
                writer_stats["overruns"] += 1
                deadline = time.monotonic()

    writer = threading.Thread(target=publish_targets, name="sdk-publisher", daemon=True)
    writer.start()
    command = np.array(args.command)
    if not np.isfinite(command).all() or abs(command[1]) > 1e-9:
        raise ValueError(
            "Use finite vx/yaw commands; this rolling interface probe supports vy=0"
        )
    started = None
    iterations = 0
    overruns = 0
    status = "failed"
    observations = []
    try:
        wait_until = time.monotonic() + 20
        while state.get()[0] is None:
            if time.monotonic() > wait_until:
                raise TimeoutError("No SDK LowState received within 20 seconds")
            time.sleep(0.005)
        started = time.monotonic()
        deadline = started
        while True:
            msg, received_at, _ = state.get()
            if time.monotonic() - received_at > 0.25:
                raise TimeoutError("Simulated LowState is stale")
            if writer_stats["error"]:
                raise RuntimeError(writer_stats["error"])
            if args.duration > 0 and msg.tick >= round(args.duration * 1000) - 1:
                status = "completed"
                break
            q, dq = policy.step(msg, command)
            target = command_message(q, dq, policy.cfg["kp"], policy.cfg["kd"])
            outgoing.put(target)
            if iterations < 100:
                observations.append(policy.history.reshape(-1).copy())
            iterations += 1
            deadline += POLICY_DT
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                overruns += 1
                deadline = time.monotonic()
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        stopped.set()
        writer.join(timeout=2)
        # A stopped Python process must not retain nonzero wheel targets.
        stop = command_message(
            np.r_[policy.nominal, np.zeros(4)],
            np.zeros(16),
            policy.cfg["kp"],
            policy.cfg["kd"],
        )
        for _ in range(3):
            publisher.Write(stop)
            time.sleep(0.01)
        subscriber.Close()
        publisher.Close()
        latency = np.array(policy.latencies)
        report = {
            "status": status,
            "policy_kind": policy.cfg["policy_kind"],
            "device": str(policy.device),
            "trained_policy": policy.cfg["policy_kind"] == "trained_him",
            "iterations": iterations,
            "received_states": state.get()[2],
            "overruns": overruns,
            "inference_ms": {
                str(p): float(np.percentile(latency, p)) for p in [50, 95, 99, 100]
            }
            if len(latency)
            else {},
            "command": command.tolist(),
            "policy_sha256": hashlib.sha256(policy.path.read_bytes()).hexdigest(),
            "domain": args.domain,
            "policy_hz": 1 / POLICY_DT,
            "command_publish_hz": 1 / SDK_DT,
            "command_writes": writer_stats["writes"],
            "writer_overruns": writer_stats["overruns"],
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        if observations:
            np.savez_compressed(
                args.report.with_suffix(".npz"), observations=np.array(observations)
            )
        print("CONTROLLER", json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--metadata", type=Path)
    p.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    p.add_argument("--domain", type=int, default=71)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--command", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()
    if not np.isfinite(args.duration) or args.duration < 0:
        p.error("duration must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
