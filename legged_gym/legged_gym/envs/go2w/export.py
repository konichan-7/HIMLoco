"""Export a Go2W checkpoint and metadata on the Isaac Gym training machine."""
import argparse
import hashlib
import json
from pathlib import Path

import isaacgym  # Isaac Gym must be imported before torch.
import numpy as np
import torch

from legged_gym.envs.go2w.go2w_config import Go2wRoughCfg, Go2wRoughCfgPPO, go2w_policy_metadata
from legged_gym.utils.helpers import class_to_dict, export_policy_as_jit
from rsl_rl.modules import HIMActorCritic


def export_go2w_policy(actor_critic, env_cfg, output, observations):
    output = Path(output)
    observations = torch.as_tensor(observations, dtype=torch.float32, device='cpu')
    if observations.ndim != 2 or observations.shape[1] != env_cfg.env.num_observations:
        raise ValueError('Expected a batch of 318-dimensional Go2W history observations')
    if not torch.isfinite(observations).all() or len(observations) == 0:
        raise ValueError('Export samples must be nonempty and finite')
    actor_critic = actor_critic.to('cpu').eval()
    export_policy_as_jit(actor_critic, str(output))
    policy_path = output / 'policy.pt'
    exported = torch.jit.load(str(policy_path), map_location='cpu').eval()
    with torch.no_grad():
        actions = actor_critic.act_inference(observations)
        exported_actions = exported(observations)
    if not torch.isfinite(actions).all() or not torch.isfinite(exported_actions).all():
        raise ValueError('Export produced nonfinite actions')
    error = float((actions - exported_actions).abs().max())
    if error > 1e-5:
        raise ValueError(f'Export parity error {error} exceeds 1e-5')
    np.savez_compressed(
        output / 'parity_samples.npz', observations=observations.numpy(),
        actor_actions=actions.numpy(), exported_actions=exported_actions.numpy(),
    )
    metadata = go2w_policy_metadata(env_cfg)
    metadata.update(policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest(),
                    parity_sample_count=len(observations), parity_max_abs_error=error)
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return policy_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', help='NPZ with real replay samples in the observations array')
    args = parser.parse_args()
    cfg, ppo = Go2wRoughCfg(), Go2wRoughCfgPPO()
    model = HIMActorCritic(cfg.env.num_observations, cfg.env.num_privileged_obs,
                          cfg.env.num_one_step_observations, cfg.env.num_actions,
                          **class_to_dict(ppo.policy))
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.samples:
        with np.load(args.samples) as samples:
            observations = samples['observations'].copy()
    else:
        observations = torch.randn(256, cfg.env.num_observations, generator=torch.Generator().manual_seed(0))
    path = export_go2w_policy(model, cfg, args.output, observations)
    metadata_path = path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text())
    metadata.update(checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
                    parity_sample_source='replay' if args.samples else 'synthetic')
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Exported {path}; parity error {metadata["parity_max_abs_error"]:.8g}')


if __name__ == '__main__':
    main()
