# Lerunbot: Lerobot Runs - RL Training System

This directory contains the Reinforcement Learning (RL) training system for **Lerunbot**, an embodied AI system that gives a standard 6-DoF robotic arm the ability to walk by pushing against the ground.

The RL system is built upon [TorchRL](https://github.com/pytorch/rl) and supports both simulated (MuJoCo) and real-world training environments.

## Training Pipeline Overview

As outlined in the technical report, the training process involves the following three main stages:

### 1. Simulation Training (SAC)
We use Soft Actor-Critic (SAC) to learn the base control policy in a MuJoCo simulation environment (`MoveBase-Lerunbot-v0`). The simulation environment mimics the physical setup (mass, friction, wheeled base) and relies purely on proprioception.
- The action space is restricted to joints 0, 1, 2, and 3.
- The default configuration trains for 1,000,000 steps using parallel environments.

**How to run:**
`train.py` reads `local_config.yaml` by default. Before running, copy the SAC configuration to `local_config.yaml`:
```bash
cp sac_config.yaml local_config.yaml
python train.py
```

### 2. Behavior Cloning (BC) for Open-loop Neural Control
To deploy the policy without a real-time feedback loop, we train an open-loop network. First, we roll out the SAC policy in simulation using `collect_checkpoint_data.py` to collect a dataset of demonstration trajectories. We then use Behavior Cloning (BC) to train a network to output complete trajectory plans based solely on the target observation.

**How to run:**
1. Collect the dataset from your SAC checkpoint:
```bash
python collect_checkpoint_data.py --checkpoint <path_to_sac_checkpoint> --mode-frames <num> --random-frames <num>
```
2. Copy the SAC configuration to `local_config.yaml` and set `demonstration.path` to the collected dataset. Then run:
```bash
cp sac_config.yaml local_config.yaml
python bc.py
```

### 3. Real-world RL (PPO)
We further improve the open-loop network directly in the real world (`Lerunbot-v0` environment) using Proximal Policy Optimization (PPO). The policy network is initialized from the BC stage, and the value network is fitted to data collected by the BC policy in simulation to speed up real-world exploration.
- Ensure the hardware components (`so101_controller`, `position_server`) are running on the robot.
- Configure `<your_robot_ip>` in `ppo_config.yaml`.
- Set `agent.load_path` in `ppo_config.yaml` to the path of your BC checkpoint (e.g. `checkpoint-stepX/agent`).

**How to run:**
Copy the PPO configuration to `local_config.yaml` before running:
```bash
cp ppo_config.yaml local_config.yaml
python train.py
```

## Configuration Files

The repository uses Hydra for configuration management. The primary configs are:
- `sac_config.yaml`: Settings for SAC simulation training.
- `ppo_config.yaml`: Settings for real-world PPO training.
- `local_config.yaml` / `sac2ppo_config.yaml`: Various configurations for BC and value network initialization.

You can override variables from the command line, for example:
```bash
python train.py --config-name sac_config experiment.train_device=cuda
```
