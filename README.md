# Micro-VLA with RL Residual Post-Training

This project implements a Vision-Language-Action (VLA) model architecture capable of residual post-training using Reinforcement Learning (RL), specifically PPO, in simulation environments like ManiSkill.

The core idea is to combine a frozen, large pre-trained VLA model (for base actions) with a lightweight, learnable residual policy (MLP) trained via RL to correct the VLA's actions for specific tasks.

## Installation & Environment Setup

1. Create a Conda environment and install PyTorch (with MPS support for Mac):
   ```bash
   conda create -n micro-vla python=3.10 -y
   conda activate micro-vla
   conda install pytorch torchvision torchaudio -c pytorch -y
   ```

2. Install ManiSkill, Gymnasium, and other dependencies:
   ```bash
   pip install mani_skill gymnasium numpy wandb
   ```

## Getting Started

### 1. Visualizing the Environment
Before training, you can run a random rollout to ensure the environment, observations (RGB-D), and video recording are working correctly.
```bash
python scripts/visualize.py
```
This will create a `videos/` folder with a sample rollout.

### 2. Training the Residual Policy (PPO)
To start RL training using the PPO algorithm in the ManiSkill environment (`PushCube-v1` by default):
```bash
python scripts/train_residual.py
```
- This script uses Weights & Biases (wandb) for logging (if configured) or basic prints.
- It runs 8 parallel environments using `AsyncVectorEnv`.
- Checkpoints will be saved periodically in the `checkpoints/` directory as `vla_policy_update_X.pt`.

### 3. Evaluation & Inference
Once you have trained the policy, you can evaluate its performance and record a video of it executing a task.
```bash
python scripts/eval_vla.py --weights checkpoints/vla_policy_update_50.pt --env PushCube-v1
```
- Look for the resulting MP4 files in the `eval_videos/` directory.

## Project Structure Highlights
- `src/agents/ppo.py`: Contains the `PPOAgent` and `RolloutBuffer` implementations.
- `src/models/residual.py`: Defines the `MicroVLAPolicy` (VLA encoder + State MLP + Actor/Critic heads).
- `src/envs/maniskill_wrapper.py`: Gymnasium wrapper to extract visual and proprioceptive states.
- `scripts/`: Entry points for training, evaluation, and visualization.
