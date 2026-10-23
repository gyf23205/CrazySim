"""Plain PPO with an MLP policy: sequential fine-tuning and training from scratch."""

from __future__ import annotations

import os

from .base import BEST_CHECKPOINT, FINAL_CHECKPOINT, Method


class Scratch(Method):
    """Every task starts from a new policy. Reference for how much the other methods transfer."""

    name = "scratch"
    reuses_policies = False

    def __init__(self, *args, checkpoint: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.checkpoint = checkpoint
        """Agent checkpoint loaded at the start of the first stage."""

    def start_stage(self, index, task, stage_dir):
        agent = self.build_ppo_agent(self.new_mlp_actor(), stage_dir)
        if self.checkpoint and index == 0:
            print(f"[INFO] Initializing from checkpoint: {self.checkpoint}")
            agent.load(self.checkpoint)
        return agent

    def end_stage(self, index, task, stage_dir, agent):
        return self.ppo_policy(agent)


class Finetune(Scratch):
    """Every task starts from the best agent (policy, value, optimizer) of the previous task.

    "Best" is skrl's best checkpoint: the highest mean training episode return, checked every
    ``methods.finetune.best_checkpoint_interval`` iterations. The best agent is also the stage's policy.
    """

    name = "finetune"

    def start_stage(self, index, task, stage_dir):
        interval = self.method_cfg["best_checkpoint_interval"] * self.agent_cfg["agent"]["rollouts"]
        agent = self.build_ppo_agent(self.new_mlp_actor(), stage_dir, checkpoint_interval=interval)
        if self.checkpoint:
            print(f"[INFO] Initializing from checkpoint: {self.checkpoint}")
            agent.load(self.checkpoint)
        return agent

    def end_stage(self, index, task, stage_dir, agent):
        best = agent.checkpoint_best_modules
        self.checkpoint = os.path.join(stage_dir, BEST_CHECKPOINT)
        if not os.path.isfile(self.checkpoint):  # no complete episode was recorded
            self.checkpoint = os.path.join(stage_dir, FINAL_CHECKPOINT)
        agent.load(self.checkpoint)
        self.stage_info = {"carried_checkpoint": self.checkpoint, "best_timestep": best["timestep"],
                           "best_training_return": float(best["reward"])}
        print(f"[INFO] Best checkpoint at timestep {best['timestep']} (training return {best['reward']:.2f})")
        return self.ppo_policy(agent)
