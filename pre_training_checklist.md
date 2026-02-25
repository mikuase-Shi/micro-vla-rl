# Pre-Training Checklist

Before launching a full-scale or long-running training job, verify the following components to ensure you haven't missed anything from the intended project architecture.

## ✅ Currently Implemented & Ready
- **PPO RL Agent**: `src/agents/ppo.py` is fully implemented with advantage estimation and surrogate loss calculation.
- **Residual Model**: `src/models/residual.py` contains the `MicroVLAPolicy` architecture.
- **ManiSkill Environment**: `src/envs/maniskill_wrapper.py` exists to format observations into RGB and State dictionaries.
- **Training Pipeline**: `scripts/train_residual.py` effectively ties the environment, buffer, and agent together for PPO training.
- **Evaluation/Visualization**: `scripts/eval_vla.py` and `scripts/visualize.py` are set up to record policy behavior.

## ✅ Currently Implemented & Ready
- **PPO RL Agent**: `src/agents/ppo.py` is fully implemented with advantage estimation and surrogate loss calculation.
- **Residual Model**: `src/models/residual.py` contains the `MicroVLAPolicy` architecture.
- **ManiSkill Environment**: `src/envs/maniskill_wrapper.py` exists to format observations into RGB and State dictionaries.
- **Training Pipeline**: `scripts/train_residual.py` effectively ties the environment, buffer, and agent together for PPO training, fully integrated with Hydra configuration system.
- **Evaluation/Visualization**: `scripts/eval_vla.py` and `scripts/visualize.py` are set up to record policy behavior.
- **SAC Agent Implementation**: `src/agents/sac.py` and `configs/agent/sac.yaml` implemented.
- **Libero Environment Integration**: `src/envs/libero_wrapper.py` and `configs/env/libero.yaml` added.
- **VLA Wrapper / Pre-trained Checkpoints**: `src/models/vla_wrapper.py` modernized to optionally load real VLA architectures (like OpenVLA variants) via HuggingFace `transformers`.
- **Hydra Configuration System**: Integrated successfully into `scripts/train_residual.py` with `configs/train.yaml`.
- **Dataset Storage**: `data/` directory confirmed.

Make sure to address these gaps if they are critical to your current experimental phase!
