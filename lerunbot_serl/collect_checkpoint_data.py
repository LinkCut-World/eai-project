import tqdm
import torch
import argparse
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from tensordict import TensorDict
from torchrl.envs.utils import ExplorationType, set_exploration_type

from lerunbot_serl.agent import make_agent
from lerunbot_serl.env import make_environment, make_single_environment


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=str, required=True, help="Path to the checkpoint directory.")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference.")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save the data file.")
    parser.add_argument("--mode-frames", "-n1", type=int, required=True, help="Number of frames to collect in mode.")
    parser.add_argument("--random-frames", "-n2", type=int, required=True, help="Number of random frames to collect.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)

    if args.output is None:
        args.output = checkpoint_path / "collected_data.pt"

    cfg = OmegaConf.load(checkpoint_path / "config.yaml")

    if args.cpu:
        cfg.experiment.train_device = "cpu"
        cfg.experiment.rollout_device = "cpu"

    # Set seeds
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    collect_env = make_environment(cfg)
    single_collect_env = make_single_environment(cfg)

    # Create agent
    agent = make_agent(
        cfg,
        cfg.experiment.train_device,
        observation_spec=single_collect_env.observation_spec,
        action_spec=single_collect_env.action_spec,
    )
    agent.loads(checkpoint_path / "agent")

    def collect(num_frames: int):
        all_data = []
        with torch.no_grad(), tqdm.tqdm(total=num_frames, unit="frames") as pbar:
            while pbar.n < num_frames:
                rollout = collect_env.rollout(
                    cfg.env.max_steps,
                    agent.policy(),
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                assert rollout[:, -1]["next", "done"].all().item(), "Every rollout should end by touching the same max steps."
                # s
                data = TensorDict(
                    {k: rollout[:, 0][k] for k in single_collect_env.observation_spec.keys()},
                    batch_size=rollout.shape[0],
                )
                # a
                data["action"] = rollout["action"].reshape(rollout.shape[0], -1)
                # s'
                data["next"] = TensorDict({k: rollout[:, -1]["next"][k] for k in single_collect_env.observation_spec.keys()})
                # r
                data["next", "reward"] = rollout["next", "reward"].sum(dim=1)
                # d
                data["next", "done"] = rollout[:, -1]["next", "done"]
                data["next", "terminated"] = rollout[:, -1]["next", "terminated"]
                all_data.append(data)
                pbar.update(data.shape[0])
        return all_data

    all_data = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        all_data += collect(args.mode_frames)
    with set_exploration_type(ExplorationType.RANDOM):
        all_data += collect(args.random_frames)
    all_data = torch.cat(all_data, dim=0)
    torch.save(all_data, args.output)
    print(all_data)
    print(f"Saved collected data to {args.output}")
