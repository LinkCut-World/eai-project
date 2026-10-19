# Lerunbot: Lerobot Runs

**Authors:** Zeyang Zhang · Chengwei Liang · Yiqi Dong

Lerunbot is an embodied AI system that enables a standard 6-DoF robotic arm to walk on an unpowered base by pushing against the ground with its end-effector.

For the full technical details (hardware, training pipeline, and experiments), please refer to our Technical Report (link to be released).

---

## Repository Structure

This repository is organized as a set of relatively independent modules:

- `gym_lowcostrobot/`: MuJoCo simulation environments and assets.
- `lerunbot/`: Real-world hardware integration and the Gymnasium environment wrapper.
  - `lerunbot/so101_controller/`: SO-101 arm serial interface.
  - `lerunbot/camera/`: Camera utilities and the pose estimation server.
- `lerunbot_serl/`: RL training code (SAC / BC / PPO) built on TorchRL.
- `vlm_control/`: VLM-based high-level control (OpenAI-compatible API client).
- `webui_control/`: Browser-based teleoperation controller.
- `examples/`: Example shell scripts to start different stacks.
- `tasks/`: Task descriptions used by `vlm_control`.

---

## Installation

```bash
git clone https://github.com/your-username/lerunbot.git
cd lerunbot

conda create -n lerunbot python=3.10
conda activate lerunbot

pip install -r requirements.txt
pip install -e .
```

---

## Usage

Please refer to the module-level READMEs for instructions:

- Real-world setup and controllers: `lerunbot/so101_controller/README.md`, `lerunbot/camera/README.md`
- RL training (simulation/real-world): `lerunbot_serl/README.md`
- VLM controller: `vlm_control/README.md`
- WebUI controller: documentation under `webui_control/`
