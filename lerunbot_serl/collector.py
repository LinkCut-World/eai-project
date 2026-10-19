from omegaconf import OmegaConf
from torchrl.collectors import SyncDataCollector, aSyncDataCollector
from contextlib import nullcontext
from functools import partial


def flatten(td):
    return td.reshape(-1)


def GAE_flatten(adv_module, td):
    # TODO: reshape to speed up adv calculation
    td = adv_module(td)
    return flatten(td)


def make_postproc(cfg, remote_agent):
    if cfg.agent.type == "sac":
        return flatten
    elif cfg.agent.type == "ppo":
        return partial(GAE_flatten, remote_agent.adv_module)
    else:
        raise NotImplementedError


def update_policy_weights(collector, agent):
    collector.update_policy_weights_(agent.policy_params())


def update_actor_critic_weights(collector, remote_agent, maybe_pause, agent):
    update_policy_weights(collector, agent)
    with maybe_pause():
        remote_agent.critic.load_state_dict(agent.critic.state_dict())


def make_update_remote_agent(cfg, collector, remote_agent):
    if cfg.agent.type == "sac":
        return partial(update_policy_weights, collector)
    elif cfg.agent.type == "ppo":
        maybe_pause = collector.pause if not cfg.experiment.sync else nullcontext
        return partial(update_actor_critic_weights, collector, remote_agent, maybe_pause)
    else:
        raise NotImplementedError


def make_collector(
    cfg,
    make_env,
    remote_agent,
    replay_buffer,
    postproc=None,
):
    if cfg.experiment.sync:
        collector = SyncDataCollector(
            create_env_fn=make_env,
            policy=remote_agent.policy(),
            postproc=postproc,
            **OmegaConf.to_container(cfg.collector.args, resolve=True),
        )
        collector.set_seed(cfg.experiment.seed)
        return collector, make_update_remote_agent(cfg, collector, remote_agent)
    else:
        collector = aSyncDataCollector(
            create_env_fn=make_env,
            policy=remote_agent.policy(),
            replay_buffer=replay_buffer,
            extend_buffer=True,
            postproc=postproc,  # use with extend_buffer=True in async collector and parallel env case. This will not require a rollout to be completed before adding data to the replay buffer.
            **OmegaConf.to_container(cfg.collector.args, resolve=True),
        )
        collector.set_seed(cfg.experiment.seed)
        collector.start()
        return collector, make_update_remote_agent(cfg, collector, remote_agent)
