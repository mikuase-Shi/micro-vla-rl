import wandb
import os

class WandbLogger:
    """
    Utility class for logging PPO training metrics to Weights & Biases.
    """
    def __init__(self, project_name="Micro-VLA-RL", run_name="ppo-run", config=None):
        self.run_name = run_name
        self.config = config if config else {}
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=self.run_name,
            config=self.config,
            monitor_gym=True,  # Automatically log gym video outputs
            save_code=True
        )
        
        # Define default metrics for nice charts
        wandb.define_metric("global_step")
        wandb.define_metric("losses/*", step_metric="global_step")
        wandb.define_metric("rollout/*", step_metric="global_step")

    def log_metrics(self, global_step, actor_loss, value_loss, entropy):
        """
        Log network losses to WandB.
        """
        wandb.log({
            "global_step": global_step,
            "losses/actor_loss": actor_loss,
            "losses/value_loss": value_loss,
            "losses/entropy": entropy
        }, step=global_step)

    def log_episode(self, global_step, episode_return, episode_length):
        """
        Log environment performance metrics to WandB.
        """
        wandb.log({
            "global_step": global_step,
            "rollout/episodic_return": episode_return,
            "rollout/episodic_length": episode_length
        }, step=global_step)

    def log_video(self, global_step, video_path):
        """
        Log an mp4 video showing the agent's performance.
        """
        if os.path.exists(video_path):
            wandb.log({
                "global_step": global_step,
                "video": wandb.Video(video_path, fps=30, format="mp4")
            }, step=global_step)

    def finish(self):
        """
        Close the WandB run.
        """
        wandb.finish()
