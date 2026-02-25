# VLA with RL Residual Post-Training: Architecture and File Structure

## 1. Overview
This project aims to implement a Vision-Language-Action (VLA) model capable of residual post-training using Reinforcement Learning (RL) in simulation environments (e.g., ManiSkill, Libero).

The core idea is:
$$ a_{final} = \pi_{VLA}(s, text) + \pi_{residual}(s, \text{vla\_features}) $$

- **VLA Policy ($\pi_{VLA}$)**: A large, pre-trained model (e.g., OpenVLA, RT-2) providing base actions. Usually frozen or fine-tuned slowly.
- **Residual Policy ($\pi_{residual}$)**: A lightweight policy (e.g., MLP) trained via RL (PPO, SAC) to correct the VLA's actions for specific tasks.

## 2. Directory Structure Recommendation

Your current structure is excellent and follows standard research engineering practices. Here is a detailed breakdown of what should go where:

```plaintext
VLARL_self/
├── README.md               # Project overview and installation instructions
├── assets/                 # Images, diagrams for README
├── configs/                # Hydra or YAML configuration files
│   ├── agent/              # RL algorithm configs (ppo.yaml, sac.yaml)
│   ├── env/                # Environment configs (maniskill.yaml, libero.yaml)
│   ├── model/              # VLA and Residual model configs
│   └── train.yaml          # Main training configuration
├── data/                   # Dataset storage (demonstrations, if any)
├── scripts/                # Entry points for training and evaluation
│   ├── train_residual.py   # Main script to start RL training
│   ├── eval_vla.py         # Script to evaluate base VLA performance
│   └── visualize.py        # Visualization of rollouts
├── src/                    # Source code package
│   ├── __init__.py
│   ├── agents/             # RL Agent implementations
│   │   ├── base_agent.py   # Abstract base class
│   │   ├── ppo.py          # PPO implementation
│   │   └── sac.py          # SAC implementation
│   ├── envs/               # Environment wrappers
│   │   ├── maniskill_wrapper.py
│   │   └── libero_wrapper.py
│   ├── models/             # Neural Network Architectures
│   │   ├── vla_wrapper.py  # Wrapper for OpenVLA/RT-X models
│   │   └── residual.py     # The residual policy network (MLP/Transformer)
│   └── utils/              # Helper functions
│       ├── logger.py       # WandB or Tensorboard logger
│       └── math_utils.py   # Tensor operations, normalization
└── tests/                  # Unit tests
```

## 3. Key Components Implementation Guide

### 3.1. `src/models/vla_wrapper.py`
*   **Purpose**: Handle loading large VLA models.
*   **Functionality**:
    *   Load weights (e.g., from HuggingFace).
    *   Provide `get_action(image, instruction)` method.
    *   Optionally provide `get_embedding(image, instruction)` if the residual policy conditions on VLA embeddings.
    *   **Optimization**: Support quantization (4-bit/8-bit) to save memory since this model is often frozen.

### 3.2. `src/models/residual.py`
*   **Purpose**: The learnable part of the system.
*   **Architecture**: Usually a multi-layer perceptron (MLP).
*   **Input**:
    *   Proprioceptive state (joint angles, gripper position).
    *   (Optional) Visual embeddings from the VLA.
*   **Output**: Action delta ($\Delta a$).
*   **Initialization**: Output should be initialized near zero to start with base VLA behavior.

### 3.3. `src/envs/wrappers.py`
*   **Purpose**: Standardize the interface between ManiSkill/Libero and your Agent.
*   **Gym/Farcama API**: Ensure `step()` returns standard `(obs, reward, done, info)`.
*   **Observation Space**: Flatten dictionaries if necessary, or keep structured for the VLA.

### 3.4. `src/agents/ppo.py` (or similar)
*   **Purpose**: Update $\pi_{residual}$.
*   **Loop**:
    1.  Get $a_{base}$ from VLA.
    2.  Get $\Delta a$ from Residual Policy.
    3.  Execute $a = a_{base} + \Delta a$.
    4.  Store tuple $(s, a, r, s')$ in buffer.
    5.  Compute advantages and update $\pi_{residual}$.

## 4. Configuration (Hydra/OmegaConf) recommended
Using a config system like prompt `hydra` allows you to easily switch between environments and models.

Example `configs/train.yaml`:
```yaml
defaults:
  - agent: ppo
  - env: maniskill_pick_cube
  - model: openvla_7b

training:
  steps: 1000000
  seed: 42
  device: "cuda:0"
```
