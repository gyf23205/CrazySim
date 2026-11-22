"""Plain PPO with an MLP policy: sequential fine-tuning and training from scratch."""

from __future__ import annotations

import torch

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


class FinetuneReset(Finetune):
    """Like :class:`Finetune`, but only the learned networks carry over; the training state is new.

    Carried from the best agent of the previous task: the policy's mean network, the value network and the
    normalizers (observation normalization, when on, and value normalization). New at every task: the action
    noise (``log_std_parameter`` back to ``initial_log_std``) and the optimizer (Adam moments, and the learning
    rate back to its configured value, as the KL-adaptive schedule adjusts it in place).
    """

    name = "finetune_reset"

    CARRIED = ("policy", "value", "state_preprocessor", "value_preprocessor")
    """Entries of skrl's agent checkpoint loaded into the new agent; ``optimizer`` is left out."""

    def start_stage(self, index, task, stage_dir):
        agent = self.build_ppo_agent(self.new_mlp_actor(), stage_dir)
        if self.checkpoint:
            print(f"[INFO] Initializing networks from checkpoint (new optimizer and action noise): {self.checkpoint}")
            saved = torch.load(self.checkpoint, map_location=self.device, weights_only=False)
            for name in self.CARRIED:
                if name not in saved:
                    continue
                state = saved[name]
                if name == "policy":
                    state = {k: v for k, v in state.items() if k != "log_std_parameter"}
                    missing, unexpected = agent.checkpoint_modules[name].load_state_dict(state, strict=False)
                    if missing != ["log_std_parameter"] or unexpected:
                        raise RuntimeError(f"Policy checkpoint mismatch: missing {missing}, unexpected {unexpected}")
                else:
                    agent.checkpoint_modules[name].load_state_dict(state)
        return agent
