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