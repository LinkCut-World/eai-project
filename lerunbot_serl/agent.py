import torch
from tensordict.nn import InteractionType, TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, ClipPPOLoss
from torchrl.objectives.sac import SACLoss
from torchrl.objectives.value.advantages import GAE
from pathlib import Path
from omegaconf import OmegaConf

from tensordict import TensorDict
from tensordict.nn import TensorDictModule


class BaseAgent:
    def policy(self) -> TensorDictModule:
        """Returns the policy module for action selection."""
        return self._policy

    def policy_params(self):
        """Returns the parameters of the policy module."""
        if not hasattr(self, "_policy_params"):
            self._policy_params = TensorDict.from_module(self.policy()).data
        return self._policy_params

    def params(self) -> TensorDict:
        """Returns the parameters of the agent, which may include learnable components beyond the policy."""
        raise NotImplementedError("Not supposed to be called.")
        # return self._params

    def dumps(self, path: Path):
        """Saves the agent's state to the specified path."""
        raise NotImplementedError

    def loads(self, path: Path):
        """Loads the agent's state from the specified path."""
        raise NotImplementedError


class BaseTrainableAgent(BaseAgent):
    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        raise NotImplementedError

    def make_optimizer(self, cfg, parameters):
        optimizer_class = getattr(torch.optim, cfg.type)
        self.optimizer = optimizer_class(parameters, **cfg[cfg.type].args)

    def make_scheduler(self, cfg):
        if cfg.type == "disabled":
            self.scheduler = None
            return
        scheduler_class = getattr(torch.optim.lr_scheduler, cfg.type)
        self.scheduler = scheduler_class(self.optimizer, **cfg[cfg.type].args)

    def dump_optimizer_and_scheduler(self, path: Path):
        """Saves states of optimizer and scheduler to the specified path."""
        path = Path(path)
        torch.save(self.optimizer.state_dict(), path / "optimizer.pt")
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), path / "scheduler.pt")

    def load_optimizer_and_scheduler(self, path: Path):
        """Loads states of optimizer and scheduler from the specified path."""
        path = Path(path)
        self.optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=self.device))
        if self.scheduler is not None:
            self.scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location=self.device))


class ConcateModule(nn.Module):
    def __init__(self, mlp, extractor):
        super().__init__()
        self.mlp = mlp
        self.extractor = extractor

    def forward(self, *args):
        x = self.mlp(*args)
        x = self.extractor(x)
        return x


class SACAgent(BaseTrainableAgent):
    def __init__(self, cfg, device, observation_spec, action_spec):
        device = torch.device(device)
        activation_class = getattr(nn, cfg.agent.sac.network.activation)

        observation_spec = observation_spec.to(device)
        action_spec = action_spec.to(device)

        in_keys_actor = list(observation_spec.keys())

        actor_mlp = MLP(
            num_cells=cfg.agent.sac.network.hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=activation_class,
            device=device,
        )
        extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.agent.sac.network.default_policy_scale}",
            scale_lb=cfg.agent.sac.network.scale_lb,
        ).to(device)

        actor_net = ConcateModule(actor_mlp, extractor)

        dist_class = TanhNormal
        dist_kwargs = {
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        }

        actor_module = TensorDictModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=[
                "loc",
                "scale",
            ],
        )
        actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=False,
        )

        # Define Critic Network
        qvalue_net = MLP(
            num_cells=cfg.agent.sac.network.hidden_sizes,
            out_features=1,
            activation_class=activation_class,
            device=device,
        )

        qvalue = ValueOperator(
            in_keys=["action"] + in_keys_actor,
            module=qvalue_net,
        )

        # init nets
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = observation_spec.rand().to(device)
            td = actor(td)
            td = qvalue(td)

        # Create SAC loss
        loss_module = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            loss_function=cfg.agent.sac.loss_function,
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=cfg.agent.sac.alpha_init,
            target_entropy=cfg.agent.sac.target_entropy,
        )
        loss_module.make_value_estimator(gamma=cfg.env.gamma)

        # Define Target Network Updater
        target_net_updater = SoftUpdate(loss_module, eps=cfg.agent.sac.target_update_polyak)

        self.device = device

        self.dist_class = dist_class
        self.dist_kwargs = dist_kwargs

        self._policy = actor
        self._params = TensorDict.from_module(loss_module).data
        self.loss_module = loss_module
        self.target_net_updater = target_net_updater

        self.make_optimizer(cfg.agent.sac.optimizer, self.loss_module.parameters())
        self.make_scheduler(cfg.agent.sac.lr_scheduler)

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        # Compute loss
        sampled_tensordict = sampled_tensordict.to(self.device)
        loss_td = self.loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        total_loss = actor_loss + q_loss + alpha_loss

        total_loss.sum().backward()
        self.optimizer.step()

        # Update qnet_target params
        self.target_net_updater.step()

        self.optimizer.zero_grad(set_to_none=True)

        loss_td.set("policy_mean", sampled_tensordict["loc"].mean())
        loss_td.set("policy_std", sampled_tensordict["scale"].mean())
        loss_td.set("action_abs_mean", sampled_tensordict["action"].abs().mean())
        loss_td.set("ground_truth_reward", sampled_tensordict["next", "reward"].mean())
        loss_td.set("td_error", torch.mean(sampled_tensordict["td_error"]))

        return loss_td.detach()

    def update_bc_only(self, sampled_tensordict: TensorDict) -> TensorDict:
        # Compute loss
        sampled_tensordict = sampled_tensordict.to(self.device)
        self.loss_module(sampled_tensordict)

        dist = self.dist_class(sampled_tensordict["loc"], sampled_tensordict["scale"], **self.dist_kwargs)
        log_prob = dist.log_prob(sampled_tensordict["action"])

        bc_loss = -log_prob.mean()
        bc_loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        loss_td = TensorDict({}, batch_size=[])
        loss_td.set("bc_loss", bc_loss.detach())
        loss_td.set("policy_mean", sampled_tensordict["loc"].mean())
        loss_td.set("policy_std", sampled_tensordict["scale"].mean())
        loss_td.set("action_abs_mean", sampled_tensordict["action"].abs().mean())
        loss_td.set("ground_truth_reward", sampled_tensordict["next", "reward"].mean())
        return loss_td.detach()

    def dumps(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.loss_module.state_dict(), path / "loss_module.pt")
        self.dump_optimizer_and_scheduler(path)

    def loads(self, path: Path, load_optimizer: bool = True):
        path = Path(path)
        self.loss_module.load_state_dict(torch.load(path / "loss_module.pt", map_location=self.device))
        if load_optimizer:
            self.load_optimizer_and_scheduler(path)


class PPOAgent(BaseTrainableAgent):
    def __init__(self, cfg, device, observation_spec, action_spec):
        device = torch.device(device)
        activation_class = getattr(nn, cfg.agent.ppo.network.activation)

        observation_spec = observation_spec.to(device)
        action_spec = action_spec.to(device)

        in_keys_actor = list(observation_spec.keys())

        actor_mlp = MLP(
            num_cells=cfg.agent.ppo.network.hidden_sizes,
            out_features=2 * action_spec.shape[-1],
            activation_class=activation_class,
            device=device,
        )
        extractor = NormalParamExtractor(
            scale_mapping=f"biased_softplus_{cfg.agent.ppo.network.default_policy_scale}",
            scale_lb=cfg.agent.ppo.network.scale_lb,
        ).to(device)

        actor_net = ConcateModule(actor_mlp, extractor)

        dist_class = TanhNormal
        dist_kwargs = {
            "low": action_spec.space.low,
            "high": action_spec.space.high,
            "tanh_loc": False,
        }

        actor_module = TensorDictModule(
            actor_net,
            in_keys=in_keys_actor,
            out_keys=[
                "loc",
                "scale",
            ],
        )
        actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=True,
        )

        critic_net = MLP(
            num_cells=cfg.agent.ppo.network.hidden_sizes,
            out_features=1,
            activation_class=activation_class,
            device=device,
        )

        critic = ValueOperator(
            in_keys=in_keys_actor,
            module=critic_net,
        )

        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            td = observation_spec.rand().to(device)
            td = actor(td)
            td = critic(td)

        adv_module = GAE(
            gamma=cfg.env.gamma,
            value_network=critic,
            device=device,
            vectorized=not cfg.compile.compile,
            **OmegaConf.to_container(cfg.agent.ppo.gae_args, resolve=True),
        )
        loss_module = ClipPPOLoss(
            actor_network=actor,
            critic_network=critic,
            **OmegaConf.to_container(cfg.agent.ppo.loss_args, resolve=True),
        )

        self.device = device

        self._policy = actor
        self._params = TensorDict.from_module(loss_module).data
        self.loss_module = loss_module
        self.adv_module = adv_module
        self.critic = critic

        self.make_optimizer(cfg.agent.ppo.optimizer, self.loss_module.parameters())
        self.make_scheduler(cfg.agent.ppo.lr_scheduler)

    def update(self, sampled_tensordict: TensorDict) -> TensorDict:
        sampled_tensordict = sampled_tensordict.to(self.device)
        loss_td = self.loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_objective"] + loss_td["loss_entropy"]
        critic_loss = loss_td["loss_critic"]

        (actor_loss + critic_loss).sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        loss_td.set("policy_mean", sampled_tensordict["loc"].mean())
        loss_td.set("policy_std", sampled_tensordict["scale"].mean())
        loss_td.set("action_abs_mean", sampled_tensordict["action"].abs().mean())
        loss_td.set("ground_truth_reward", sampled_tensordict["next", "reward"].mean())
        loss_td.set("action_log_prob", sampled_tensordict["action_log_prob"].mean())
        loss_td.set("advantage", sampled_tensordict["advantage"].mean())
        loss_td.set("state_value", sampled_tensordict["state_value"].mean())
        loss_td.set("value_target", sampled_tensordict["value_target"].mean())
        loss_td.set("value_error", (sampled_tensordict["state_value"] - sampled_tensordict["value_target"]).abs().mean())
        return loss_td.detach()

    def dumps(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.loss_module.state_dict(), path / "loss_module.pt")
        self.dump_optimizer_and_scheduler(path)

    def loads(self, path: Path, load_optimizer: bool = True):
        path = Path(path)
        self.loss_module.load_state_dict(torch.load(path / "loss_module.pt", map_location=self.device))
        if load_optimizer:
            self.load_optimizer_and_scheduler(path)


def make_agent(cfg, device, observation_spec, action_spec) -> BaseTrainableAgent:
    if cfg.agent.type == "sac":
        return SACAgent(
            cfg,
            device,
            observation_spec=observation_spec,
            action_spec=action_spec,
        )
    elif cfg.agent.type == "ppo":
        return PPOAgent(
            cfg,
            device,
            observation_spec=observation_spec,
            action_spec=action_spec,
        )
    else:
        raise NotImplementedError(f"Agent {cfg.agent} not implemented")
