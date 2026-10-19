from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from omegaconf import OmegaConf
from tensordict import TensorDict


def make_replay_buffer(cfg, fake_data: TensorDict):
    if cfg.replay_buffer.prioritized:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(**OmegaConf.to_container(cfg.replay_buffer.storage_args, resolve=True)),
            sampler=PrioritizedSampler(**OmegaConf.to_container(cfg.replay_buffer.prioritized_sampler_args, resolve=True)),
            priority_key="td_error",
            **OmegaConf.to_container(cfg.replay_buffer.args, resolve=True),
        )
    else:
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(**OmegaConf.to_container(cfg.replay_buffer.storage_args, resolve=True)),
            shared=not cfg.experiment.sync,  # whether the buffer will be shared using multiprocessing or not
            **OmegaConf.to_container(cfg.replay_buffer.args, resolve=True),
        )
    replay_buffer.extend(fake_data)
    replay_buffer.empty()
    return replay_buffer
