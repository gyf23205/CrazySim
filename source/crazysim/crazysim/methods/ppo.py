"""Plain PPO with an MLP policy: sequential fine-tuning and training from scratch."""

from __future__ import annotations

from .base import Method


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
        self.load_best_checkpoint(agent, stage_dir)
        return self.ppo_policy(agent)

    def resume(self, stages):
        pass


class Finetune(Scratch):
    """Every task starts from the best agent (policy, value, optimizer) of the previous task."""

    name = "finetune"

    def start_stage(self, index, task, stage_dir):
        agent = self.build_ppo_agent(self.new_mlp_actor(), stage_dir)
        if self.checkpoint:
            print(f"[INFO] Initializing from checkpoint: {self.checkpoint}")
            agent.load(self.checkpoint)
        return agent

    def end_stage(self, index, task, stage_dir, agent):
        self.checkpoint = self.load_best_checkpoint(agent, stage_dir)
        return self.ppo_policy(agent)

    def resume(self, stages):
        # "carried_checkpoint": sessions made before every method selected the best checkpoint
        last = stages[-1]
        self.checkpoint = last.get("selected_checkpoint", last.get("carried_checkpoint"))
