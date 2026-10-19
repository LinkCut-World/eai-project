from __future__ import annotations

import time

import warnings
import shutil
import json
from functools import partial

import math
import hydra
import numpy as np
import tensordict
import torch
import torch.cuda
import tqdm
import copy
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.envs import make_composite_from_td
import logging
import wandb
from omegaconf import OmegaConf, open_dict
from pathlib import Path
from contextlib import nullcontext

from lerunbot_serl.env import make_environment, make_single_environment
from lerunbot_serl.collector import make_postproc, make_collector
from lerunbot_serl.replaybuffer import make_replay_buffer
from lerunbot_serl.agent import make_agent

torch.set_float32_matmul_precision("high")
tensordict.nn.functional_modules._exclude_td_from_pytree().set()
REALWORLD = False


def log_to_wandb(wandb_run, step, metrics, prefix=""):
    prefixed = {f"{prefix}/{k}": float(v) for k, v in metrics.items()}
    if wandb_run is None:
        return
    try:
        wandb_run.log(prefixed, step)
    except Exception as e:
        torchrl_logger.warning(f"Failed to log to wandb at step {step}: {e}")


def override(cfg_, override_dict):
    for key, o in override_dict.items():
        if o["type"] == "recursive":
            override(cfg_[key], o["value"])
        elif o["type"] == "override":
            cfg_[key] = o["value"]
        else:
            raise NotImplementedError(f"Override type {o['type']} not implemented")


@hydra.main(version_base="1.1", config_path="", config_name="local_config")
def main(cfg):  # noqa: F821
    # init resume checkpoint path
    resume_path = None
    if cfg.resume.mode == "specified":
        resume_path = Path(cfg.resume.specified_path)
    elif cfg.resume.mode != "disabled":
        raise NotImplementedError(f"Resume mode {cfg.resume.mode} not implemented")

    if resume_path is not None:
        print(f"Resuming from checkpoint: {resume_path}")
        print("Loading config from checkpoint")
        cfg_ = OmegaConf.load(resume_path / "config.yaml")
        override_dict = OmegaConf.to_container(cfg.resume.overrides)
        override(cfg_, override_dict)
        cfg = cfg_
    else:
        print("loading config from local_config.yaml")
    global REALWORLD
    REALWORLD = cfg.experiment.realworld

    # logging setup
    log_path = cfg.training.logging.path
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    torchrl_logger.addHandler(fh)

    start_step = 0
    if resume_path is not None:
        metadata = json.load(open(resume_path / "metadata.json", "r", encoding="utf-8"))
        start_step = metadata["step"] + 1

    # Wandb setup
    wandb_run, wandb_run_id = None, None
    logging_type = cfg.training.logging.type
    if logging_type == "wandb":
        if resume_path is None:
            wandb_run = wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                **OmegaConf.to_container(cfg.training.logging.wandb.args, resolve=True),
            )
            file_path = Path(__file__).resolve()
            dir_path = file_path.parent
            wandb_run.log_code(str(dir_path))
            if cfg.env.gym.name == "MoveBase-Lerunbot-v0":
                wandb_run.save(str(dir_path.parent / "gym_lowcostrobot" / "envs" / "move_base_lerunbot_env.py"))
                wandb_run.save(str(dir_path.parent / "gym_lowcostrobot" / "assets" / "soarm101" / "move_base.xml"))
            elif cfg.env.gym.name == "Lerunbot-v0":
                wandb_run.save(str(dir_path.parent / "lerunbot" / "envs" / "lerunbot_env.py"))
        else:
            wandb_run = wandb.init(
                config=OmegaConf.to_container(cfg, resolve=True),
                id=metadata["wandb_id"],
                resume="must",
                **OmegaConf.to_container(cfg.training.logging.wandb.args, resolve=True),
            )
        wandb_run_id = wandb_run.id
    elif logging_type != "disabled":
        raise NotImplementedError(f"Logging type {logging_type} not implemented")

    # Set seeds
    torch.manual_seed(cfg.experiment.seed)
    np.random.seed(cfg.experiment.seed)

    # Create environments
    collect_env = make_environment(cfg)
    single_collect_env = make_single_environment(cfg)

    torchrl_logger.info(f"Observation spec: {single_collect_env.observation_spec}")
    torchrl_logger.info(f"Action spec: {single_collect_env.action_spec}")

    # Create agent
    agent = make_agent(
        cfg,
        cfg.experiment.train_device,
        observation_spec=single_collect_env.observation_spec,
        action_spec=single_collect_env.action_spec,
        # fake_batch=eval_env.fake_tensordict(),
    )
    remote_agent = make_agent(
        cfg,
        cfg.experiment.rollout_device,
        observation_spec=single_collect_env.observation_spec,
        action_spec=single_collect_env.action_spec,
    )
    if cfg.agent.load_path is not None:
        torchrl_logger.info(f"Loading agent from {cfg.agent.load_path}")
        agent.loads(cfg.agent.load_path, load_optimizer=cfg.agent.load_optimizer)
    if resume_path is not None:
        torchrl_logger.info(f"Resuming agent from {resume_path / 'agent'}")
        agent.loads(resume_path / "agent")

    # Create postproc
    postproc = make_postproc(cfg, remote_agent)

    # with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
    #     eval_rollout = collect_env.rollout(
    #         cfg.training.evaluation.max_steps,
    #         agent.policy(),
    #         auto_reset=True,
    #         auto_cast_to_device=True,
    #         break_when_any_done=True,
    #     )
    #     print(eval_rollout)
    # return

    if cfg.compile.compile:
        # This function wraps :func:`~torch.compile` to add a warm-up phase. During the warm-up phase, the original model is used. After the warm-up phase, the model is compiled using `torch.compile`.
        agent.update = compile_with_warmup(agent.update, mode=cfg.compile.compile_mode, warmup=2)
        if cfg.demonstration.path is not None and cfg.demonstration.bc_loss_end:
            agent.update_with_demo_mask = compile_with_warmup(agent.update_with_demo_mask, mode=cfg.compile.compile_mode, warmup=2)
    if cfg.compile.cudagraphs:
        # See https://docs.pytorch.ac.cn/tensordict/stable/reference/generated/tensordict.nn.CudaGraphModule.html
        agent.update = CudaGraphModule(agent.update, in_keys=[], out_keys=[], warmup=10)
        if cfg.demonstration.path is not None and cfg.demonstration.bc_loss_end:
            agent.update_with_demo_mask = CudaGraphModule(agent.update_with_demo_mask, in_keys=[], out_keys=[], warmup=10)

    # Create fake data for replay buffer initialization
    with torch.no_grad():
        fake_data = collect_env.reset()
        fake_data = remote_agent.policy()(fake_data)
        fake_data = collect_env.step(fake_data)
        fake_data = fake_data.unsqueeze(1)
        fake_data = postproc(fake_data)
    torchrl_logger.info(f"Example data in replay buffer: {fake_data}")

    # Create replay buffer
    replay_buffer = make_replay_buffer(cfg, fake_data)
    resume_write_count = 0
    if resume_path is not None:
        replay_buffer.loads(resume_path / "replay_buffer")
        resume_write_count = metadata["write_count"]

    # Fill replay buffer with random data if needed
    if resume_path is None and cfg.collector.init_random_frames > 0:
        torchrl_logger.info("Filling replay buffer with random data")
        data_ = collect_env.reset()
        target = cfg.collector.init_random_frames
        current = 0
        with tqdm.tqdm(total=target, desc="Filling replay buffer", unit="frames") as pbar_fill:
            pbar_fill.update(current)
            rollout = []
            while current < target:
                # Here we use random actions so it may have keys different from that obtained from policy. For example, action_log_prob may not be present. This is not expected.
                collect_env.rand_action(data_)
                data, data_ = collect_env.step_and_maybe_reset(data_)
                rollout.append(data)

                delta = data.numel()
                pbar_fill.update(delta)
                current += delta
            rollout = TensorDict.stack(rollout, dim=1)
            rollout = postproc(rollout)
            replay_buffer.extend(rollout)

    # Create data collector
    collector, update_remote_agent = make_collector(
        cfg,
        make_env=partial(make_environment, cfg),
        remote_agent=remote_agent,
        replay_buffer=replay_buffer,
        postproc=postproc,
    )
    update_remote_agent(agent)

    # Main loop
    sync = cfg.experiment.sync
    num_steps = cfg.training.num_steps
    eval_freq = cfg.training.evaluation.freq
    save_freq = cfg.training.save.freq
    log_freq = cfg.training.logging.freq

    pbar = tqdm.tqdm(total=num_steps)

    # Wait till we have enough data to start training
    if not sync:
        while resume_write_count + replay_buffer.write_count <= cfg.replay_buffer.args.batch_size:
            time.sleep(0.01)

    # sampled_tensordict = replay_buffer.sample()
    # print(sampled_tensordict)
    # for k, v in sampled_tensordict[0].items():
    #     if k != "next":
    #         print(f"{k}: {v}")
    #     else:
    #         for k2, v2 in v.items():
    #             print(f"  next/{k2}: {v2}")
    # collector.async_shutdown()
    # return

    # eval_env = make_environment(cfg, eval=True)

    if sync:
        collector_iter = iter(collector)
        next(collector_iter)  # warm-up
    else:
        # TODO: resume version information
        # TODO: write count in async resume
        if resume_path is not None:
            raise NotImplementedError("Resuming async training not implemented yet.")
        off_policy_tolerance = cfg.training.off_policy_tolerance
        if off_policy_tolerance == 0:
            raise NotImplementedError("off_policy_tolerance of 0 is not supported.")
        per_version_frames = len(replay_buffer.storage) // off_policy_tolerance if off_policy_tolerance else 0
        last_version_write_count = 0

    example_sampled_tensordict_printed = False
    metrics, episode_rewards = [], []
    eval_discounted_reward_sums = []

    torchrl_logger.info("\n" * 5)
    torchrl_logger.info("Starting main training loop")
    torchrl_logger.info("\n" * 5)
    for i in range(num_steps):
        if i < start_step:
            pbar.update(1)
            continue

        timeit.printevery(
            num_prints=num_steps // log_freq,
            total_count=num_steps,
            erase=True,
        )

        if sync:
            if (i % cfg.collector.collect_freq) == 0:
                update_remote_agent(agent)
                with timeit("collect"):
                    tensordict = next(collector_iter)
                with timeit("rb - extend"):
                    replay_buffer.extend(tensordict)
        else:
            if (i % cfg.collector.update_freq) == 0:
                if replay_buffer.write_count < last_version_write_count + per_version_frames:
                    torchrl_logger.info(f"Waiting for replay buffer to have enough new data. {last_version_write_count + per_version_frames - replay_buffer.write_count} frames needed.")
                    while replay_buffer.write_count < last_version_write_count + per_version_frames:
                        time.sleep(0.01)

                torchrl_logger.info("Updating remote agent")
                update_remote_agent(agent)
                last_version_write_count = replay_buffer.write_count

        pbar.update(1)

        # Optimization steps
        with timeit("train"):
            with timeit("train - rb - sample"):
                sampled_tensordict = replay_buffer.sample()
            with timeit("train - update"):
                metric = agent.update(sampled_tensordict)
                if sync and cfg.replay_buffer.prioritized:
                    replay_buffer.update_tensordict_priority(sampled_tensordict)
                if not example_sampled_tensordict_printed:
                    torchrl_logger.info(f"Example sampled tensordict after update: {sampled_tensordict}")
                    example_sampled_tensordict_printed = True
                if agent.scheduler is not None:
                    agent.scheduler.step()
                    metric.set("lr", agent.scheduler.get_last_lr()[0])

                episode_rewards.append(sampled_tensordict["next", "episode_reward"].reshape(-1)[sampled_tensordict["next", "done"].reshape(-1).bool()])
                metrics.append(metric)

        log_step = i if cfg.training.logging.count == "training_steps" else resume_write_count + replay_buffer.write_count

        # Eval
        if i == 0 or (i % eval_freq) == (eval_freq - 1):
            torchrl_logger.info("Evaluating")
            maybe_pause = collector.pause() if not sync else nullcontext()
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad(), timeit("eval"), maybe_pause:
                eval_env = make_environment(cfg, eval=True)
                if REALWORLD:
                    eval_env.reset(stop_skip=True)  # special for Lerunbot-v0 env, because Lerunbot-v0 by default skips the first 10 steps which are mostly used by torchrl to extract observation_spec and action_spec, initialize replay buffer, etc. This causes the first 10 steps of every episode to be skipped during evaluation, which is undesirable. By calling reset with stop_skip=True, we can avoid this issue.
                steps = math.ceil(cfg.training.evaluation.frames / eval_env.num_workers)
                eval_rollout = eval_env.rollout(
                    steps,
                    agent.policy(),
                    auto_reset=True,
                    auto_cast_to_device=True,
                    break_when_any_done=False,
                    break_when_all_done=False,
                )  # [B, T, ...]
                nexts = eval_rollout["next"]

                eval_metrics = {}

                nexts_sum = TensorDict({}, batch_size=nexts.batch_size)
                for key in nexts.keys():
                    if not isinstance(nexts[key], torch.Tensor):
                        continue
                    if nexts[key].numel() != nexts.numel():
                        continue
                    nexts[key] = nexts[key].reshape(nexts.batch_size)
                    eval_metrics[f"mean_{key}"] = nexts[key].float().mean().item()
                    nexts_sum.set(f"{key}_sum", nexts[key].clone().float())
                discounted_reward_sum = nexts["reward"].clone()

                for t in reversed(range(nexts.shape[1] - 1)):
                    nexts_sum[:, t] += (1 - nexts["done"][:, t].float()) * nexts_sum[:, t + 1]
                    discounted_reward_sum[:, t] += (1 - nexts["done"][:, t].float()) * cfg.env.gamma * discounted_reward_sum[:, t + 1]
                nexts_sum.set("discounted_reward_sum", discounted_reward_sum)

                mask = eval_rollout["is_init"].reshape(nexts.batch_size).bool()
                eval_metrics["eval_episode_count"] = int(mask.sum().item())
                for key in nexts_sum.keys():
                    eval_metrics[key] = nexts_sum[key][mask].float().mean().item()

                log_to_wandb(wandb_run, log_step, eval_metrics, "eval")
                torchrl_logger.info(f"Eval metrics: {eval_metrics}")

                eval_discounted_reward_sums.append(eval_metrics["discounted_reward_sum"])
                eval_env.close()

        # Save
        if (i % save_freq) == (save_freq - 1):
            torchrl_logger.info("Saving checkpoint")
            save_path = Path(cfg.experiment.checkpoint_root) / f"checkpoint-step{i}"
            if save_path.exists():
                shutil.rmtree(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            agent.dumps(save_path / "agent")
            replay_buffer.dumps(save_path / "replay_buffer")
            with open(save_path / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "step": i,
                        "write_count": resume_write_count + replay_buffer.write_count,
                        "wandb_id": wandb_run_id,
                    },
                    f,
                )
            OmegaConf.save(cfg, save_path / "config.yaml")

            torchrl_logger.info(f"Saved checkpoint to {save_path}")

        # Logging
        if (i % log_freq) == (log_freq - 1):
            torchrl_logger.info("Logging")
            collected_frames = resume_write_count + replay_buffer.write_count
            metrics_m = torch.stack(metrics).mean()
            metrics_m["collected_frames"] = collected_frames

            episode_rewards = torch.cat(episode_rewards)
            if len(episode_rewards) > 0:
                metrics_m["ground_truth_episolde_reward"] = episode_rewards.mean().item()
                metrics_m["ground_truth_episolde_reward_max"] = episode_rewards.max().item()

            log_to_wandb(wandb_run, log_step, metrics_m, "train")
            log_to_wandb(wandb_run, log_step, timeit.todict(), "time")
            log_to_wandb(wandb_run, log_step, {"speed": pbar.format_dict["rate"]}, "time")
            metrics, episode_rewards = [], []

    if not sync:
        collector.async_shutdown()
    if not eval_env.is_closed:
        eval_env.close()

    output = np.array(eval_discounted_reward_sums).mean() if len(eval_discounted_reward_sums) > 0 else None
    print(f"FINAL_OUTPUT: {output}")


if __name__ == "__main__":
    main()
