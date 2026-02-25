# Environment Setup for Micro-VLA

Run the following commands in your terminal to set up the environment on your Mac (ensuring PyTorch has MPS support):

```bash
# 1. Create a new Conda environment with Python 3.10
conda create -n micro-vla python=3.10 -y

# 2. Activate the environment
conda activate micro-vla

# 3. Install PyTorch (The default PyTorch builds for macOS already include MPS support)
conda install pytorch torchvision torchaudio -c pytorch -y

# 4. Install ManiSkill, Gymnasium, and necessary RL tracking tools
pip install mani_skill gymnasium numpy wandb
```
