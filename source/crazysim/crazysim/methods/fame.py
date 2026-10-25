"""FAME (Anand and Precup, 2023, "Prediction and Control in Continual Reinforcement Learning";
fast and meta learners), adapted to continuous actions with PPO.

Ported from ``less_leg_walking_1`` (``baselines/FAME/``, ``scripts/skrl/train_FAME.py``).

- Fast learner: PPO with a tanh-squashed Gaussian actor (state-dependent standard deviation) and an
  automatically tuned entropy temperature (as in SAC). It is re-initialized at every task.
- Meta learner: an actor of the same shape, distilled from the fast learner at the end of every task
  with a Wasserstein-like loss on the Gaussian ``(mu, std)``. Earlier knowledge is kept by also
  matching the previous meta learner on states stored from earlier tasks.
- Task switch: a fresh actor and the meta actor are both evaluated on the new task; the better one
  initializes the fast learner (actor only).

Changes from the reference: rollout and distillation buffers are tensors filled for all
environments at once (same contents: the most recent states of the task, and a FIFO of past-task
states); the advantage estimate bootstraps from the value of the next state of the same episode at a
time-out instead of from the value of the next episode's first state; and the PPO update computes
log-probabilities from the stored pre-squash samples instead of inverting the stored actions with
atanh. The inversion fails when an action saturates (``tanh`` returns exactly +-1 in float32): with a
pre-squash mean of 10 and standard deviation 0.0067 it gives a log-probability of about -84000
instead of 22, the probability ratio overflows and the parameters become NaN.

The fast learner kept from each stage is skrl's best checkpoint of the stage (highest mean training
episode return), as for the other methods. If that is not the final one, the states used for the
distillation are re-collected with the best fast actor, since the states recorded during training
were visited by later actors. The policy saved for each stage is that fast actor; the meta actor is
saved next to it as ``meta_policy.pt``.

Resuming a session (:meth:`Fame.resume`) loads the last meta actor. The buffer of earlier-task states
is not saved, so it is refilled with states visited by each stage's saved fast actor (stochastic
actions) on its task, as many per task as the distillation buffer held during training.
"""

from __future__ import annotations

import copy
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from skrl.agents.torch import Agent

from .base import Method, make_mlp

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _orthogonal_init(m: nn.Module):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


class TanhTransform(pyd.transforms.Transform):
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size: int = 1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    """Normal distribution squashed by tanh. ``mean`` is the squashed mean."""

    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        super().__init__(pyd.Normal(loc, scale), [TanhTransform()])

    @property
    def mean(self):
        return self.loc.tanh()


class SquashedActor(nn.Module):
    """MLP trunk returning ``(mu, log_std)``; actions are tanh-squashed into (-1, 1)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.trunk = make_mlp(obs_dim, hidden_dims, 2 * action_dim)
        self.apply(_orthogonal_init)

    def dist_params(self, obs):
        """``(mu, std)`` of the Gaussian before the tanh squash."""
        mu, log_std = self.trunk(obs).chunk(2, dim=-1)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (torch.tanh(log_std) + 1.0)
        return mu, log_std.exp()

    def forward(self, obs) -> SquashedNormal:
        return SquashedNormal(*self.dist_params(obs))

    def log_prob_raw(self, obs, raw_actions):
        """Log-probability of the squashed action ``tanh(raw_actions)``, computed from the pre-squash sample."""
        mu, std = self.dist_params(obs)
        log_prob = pyd.Normal(mu, std).log_prob(raw_actions) - TanhTransform().log_abs_det_jacobian(raw_actions, None)
        return log_prob.sum(-1, keepdim=True)


class SquashedMean(nn.Module):
    """Deterministic action ``tanh(mu)`` of a :class:`SquashedActor`."""

    def __init__(self, actor: SquashedActor):
        super().__init__()
        self.actor = actor

    def forward(self, obs):
        return self.actor.dist_params(obs)[0].tanh()


class StateBuffer:
    """FIFO buffer of states on the GPU; keeps the most recent ``capacity`` rows."""

    def __init__(self, capacity: int, dim: int, device):
        self.data = torch.zeros(capacity, dim, device=device)
        self.capacity = capacity
        self.index = 0
        self.size = 0

    def add(self, states: torch.Tensor):
        states = states[-self.capacity :]
        n = states.shape[0]
        idx = (self.index + torch.arange(n, device=states.device)) % self.capacity
        self.data[idx] = states
        self.index = (self.index + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> torch.Tensor:
        return self.data[torch.randint(self.size, (batch_size,), device=self.data.device)]

    def all(self) -> torch.Tensor:
        return self.data[: self.size]

    def clear(self):
        self.index = self.size = 0


class FAMEAgent(Agent):
    """skrl agent holding the fast and meta learners. It persists across stages."""

    def __init__(self, observation_space, action_space, device, cfg: dict):
        super().__init__(models={}, memory=None, observation_space=observation_space, action_space=action_space,
                         device=device, cfg=cfg)
        self.obs_dim = observation_space.shape[0]
        self.action_dim = action_space.shape[0]
        self.hidden_dims = list(cfg["hidden_dims"])
        self._reset_fast_learner()
        self.meta_actor = SquashedActor(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.checkpoint_modules["meta_actor"] = self.meta_actor
        self._meta_trained = False

        self.fast2meta = StateBuffer(cfg["size_fast2meta"], self.obs_dim, self.device)
        self.meta_buffer = StateBuffer(cfg["size_meta"], self.obs_dim, self.device)
        self._rollout = None
        self._step = 0

    # -- skrl interface ------------------------------------------------------------------------

    def init(self, trainer_cfg=None):
        """Called by every stage's trainer: opens the stage's TensorBoard writer, clears the rollout and
        restarts the episode-return tracking, so that the stage's best checkpoint is chosen among this
        stage's agents only."""
        super().init(trainer_cfg=trainer_cfg)
        self._rollout = None
        self._step = 0
        self.checkpoint_best_modules = {"timestep": 0, "reward": -(2**31), "saved": False, "modules": {}}
        self.tracking_data.clear()
        self._track_rewards.clear()
        self._track_timesteps.clear()
        self._cumulative_rewards = None
        self._cumulative_timesteps = None

    def set_running_mode(self, mode: str):
        self.training = mode == "train"

    def act(self, states, timestep, timesteps):
        with torch.no_grad():
            mu, std = self.fast_actor.dist_params(states)
            self._last_raw_actions = pyd.Normal(mu, std).sample()
            self._last_log_probs = self.fast_actor.log_prob_raw(states, self._last_raw_actions)
            self._last_values = self.fast_critic(states)
        return self._last_raw_actions.tanh(), self._last_log_probs, {"mean_actions": mu.tanh()}

    def record_transition(self, states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
        super().record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
        if self._rollout is None:
            T, N = self.cfg["rollout_steps"], states.shape[0]
            self._rollout = {
                "states": torch.zeros(T, N, self.obs_dim, device=self.device),
                "next_states": torch.zeros(T, N, self.obs_dim, device=self.device),
                "raw_actions": torch.zeros(T, N, self.action_dim, device=self.device),
                "rewards": torch.zeros(T, N, 1, device=self.device),
                "terminated": torch.zeros(T, N, 1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(T, N, 1, dtype=torch.bool, device=self.device),
                "values": torch.zeros(T, N, 1, device=self.device),
                "log_probs": torch.zeros(T, N, 1, device=self.device),
            }
        r = self._rollout
        t = self._step
        r["states"][t] = states
        r["next_states"][t] = next_states
        r["raw_actions"][t] = self._last_raw_actions
        r["rewards"][t] = rewards.view(-1, 1) * self.cfg["rewards_shaper_scale"]
        r["terminated"][t] = terminated.view(-1, 1)
        r["truncated"][t] = truncated.view(-1, 1)
        r["values"][t] = self._last_values
        r["log_probs"][t] = self._last_log_probs
        self._step += 1
        self.fast2meta.add(states)

    def post_interaction(self, timestep, timesteps):
        if self._step >= self.cfg["rollout_steps"]:
            self._update()
            self._step = 0
        super().post_interaction(timestep, timesteps)

    # -- task switch -------------------------------------------------------------------------------

    def start_task(self, task: str, evaluate, first: bool) -> str:
        """Re-initialize the fast learner and pick its initial actor. Returns the choice."""
        if first:
            return "initial"
        self._reset_fast_learner()
        steps = self.cfg["detection_steps"]
        avg_fresh = evaluate(self.fast_actor, steps)
        avg_meta = evaluate(self.meta_actor, steps)
        choice = "meta" if avg_meta > avg_fresh else "fresh"
        if choice == "meta":
            self.fast_actor.load_state_dict(copy.deepcopy(self.meta_actor.state_dict()))
            self.fast_actor_opt = Adam(self.fast_actor.parameters(), lr=self.cfg["lr_actor"])
        print(f"[FAME] Detection on {task}: fresh return {avg_fresh:.3f}, meta return {avg_meta:.3f} -> {choice}")
        self.detection = {"fresh_return": avg_fresh, "meta_return": avg_meta, "choice": choice}
        return choice

    def end_task(self):
        """Distill the fast learner into the meta learner and keep this task's states."""
        start = time.time()
        self._train_meta()
        print(f"[FAME] Meta distillation done in {time.time() - start:.1f} s")
        self.meta_buffer.add(self.fast2meta.all())
        self.fast2meta.clear()

    # -- internals ---------------------------------------------------------------------------------

    def _reset_fast_learner(self):
        cfg = self.cfg
        self.fast_actor = SquashedActor(self.obs_dim, self.action_dim, self.hidden_dims).to(self.device)
        self.fast_critic = make_mlp(self.obs_dim, self.hidden_dims, 1).to(self.device)
        self.fast_critic.apply(_orthogonal_init)
        self.fast_actor_opt = Adam(self.fast_actor.parameters(), lr=cfg["lr_actor"])
        self.fast_critic_opt = Adam(self.fast_critic.parameters(), lr=cfg["lr_critic"])
        self.target_entropy = cfg["target_entropy"] if cfg["target_entropy"] is not None else -float(self.action_dim)
        self.log_alpha = torch.tensor(math.log(cfg["init_temperature"]), device=self.device, requires_grad=True)
        self.log_alpha_opt = Adam([self.log_alpha], lr=cfg["alpha_lr"])
        self.checkpoint_modules["fast_actor"] = self.fast_actor
        self.checkpoint_modules["fast_critic"] = self.fast_critic

    def _update(self):
        cfg = self.cfg
        r = self._rollout
        T, N = r["rewards"].shape[:2]
        gamma, lam, clip = cfg["gamma"], cfg["gae_lambda"], cfg["clip_ratio"]

        with torch.no_grad():
            next_values = self.fast_critic(r["next_states"])
            not_terminated = (~r["terminated"]).float()
            not_done = (~(r["terminated"] | r["truncated"])).float()
            advantages = torch.zeros_like(r["rewards"])
            gae = torch.zeros(N, 1, device=self.device)
            for t in reversed(range(T)):
                delta = r["rewards"][t] + gamma * next_values[t] * not_terminated[t] - r["values"][t]
                gae = delta + gamma * lam * not_done[t] * gae
                advantages[t] = gae
            returns = advantages + r["values"]

        s = r["states"].view(T * N, -1)
        raw = r["raw_actions"].view(T * N, -1)
        old_lp = r["log_probs"].view(T * N, 1)
        old_v = r["values"].view(T * N, 1)
        ret = returns.view(T * N, 1)
        adv = advantages.view(T * N, 1)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        total = T * N
        mini_batch = min(cfg["mini_batch_size"], max(1, total // cfg["num_mini_batches"]))
        stats = torch.zeros(4, device=self.device)
        n_updates = 0
        for _ in range(cfg["ppo_epochs"]):
            perm = torch.randperm(total, device=self.device)
            for begin in range(0, total, mini_batch):
                idx = perm[begin : begin + mini_batch]
                if len(idx) < 2:
                    continue
                obs_b, adv_b, ret_b = s[idx], adv[idx], ret[idx]

                # clipped surrogate on the stored actions
                ratio = (self.fast_actor.log_prob_raw(obs_b, raw[idx]) - old_lp[idx]).exp()
                pg_loss = -torch.min(ratio * adv_b, ratio.clamp(1 - clip, 1 + clip) * adv_b).mean()
                # entropy regularization on fresh reparameterized samples, weighted by the temperature
                mu_b, std_b = self.fast_actor.dist_params(obs_b)
                raw_fresh = pyd.Normal(mu_b, std_b).rsample()
                logp_fresh = self.fast_actor.log_prob_raw(obs_b, raw_fresh)
                actor_loss = pg_loss + self.log_alpha.exp().detach() * logp_fresh.mean()
                self.fast_actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.fast_actor.parameters(), cfg["grad_norm_clip"])
                self.fast_actor_opt.step()

                # temperature update toward the target entropy
                alpha_loss = (self.log_alpha.exp() * (-logp_fresh - self.target_entropy).detach()).mean()
                self.log_alpha_opt.zero_grad()
                alpha_loss.backward()
                self.log_alpha_opt.step()
                with torch.no_grad():
                    self.log_alpha.clamp_(min=-5.0, max=2.0)

                # clipped value loss
                new_v = self.fast_critic(obs_b)
                v_clipped = old_v[idx] + (new_v - old_v[idx]).clamp(-clip, clip)
                vf_loss = cfg["value_coef"] * torch.max(F.mse_loss(new_v, ret_b), F.mse_loss(v_clipped, ret_b))
                self.fast_critic_opt.zero_grad()
                vf_loss.backward()
                nn.utils.clip_grad_norm_(self.fast_critic.parameters(), cfg["grad_norm_clip"])
                self.fast_critic_opt.step()

                stats += torch.stack([pg_loss, vf_loss, -logp_fresh.mean(), self.log_alpha.exp()]).detach()
                n_updates += 1

        stats = (stats / max(n_updates, 1)).tolist()
        self.track_data("Loss / Policy loss", stats[0])
        self.track_data("Loss / Value loss", stats[1])
        self.track_data("Loss / Entropy estimate", stats[2])
        self.track_data("Policy / Temperature", stats[3])
        with torch.no_grad():
            self.track_data("Policy / Standard deviation", self.fast_actor.dist_params(s)[1].mean().item())

    def _train_meta(self):
        cfg = self.cfg
        batch_size = cfg["mini_batch_size"]
        new_ready = self.fast2meta.size >= batch_size
        old_ready = self._meta_trained and self.meta_buffer.size >= batch_size
        if not new_ready and not old_ready:
            print("[FAME] Distillation buffers too small; skipping.")
            return
        previous_meta = copy.deepcopy(self.meta_actor) if old_ready else None

        optimizer = Adam(self.meta_actor.parameters(), lr=cfg["lr_meta"])
        scheduler = ExponentialLR(optimizer, gamma=cfg["meta_lr_scheduler_gamma"])
        steps = max(1, max(self.meta_buffer.size, self.fast2meta.size) // batch_size - 1)

        def wasserstein(states, teacher):
            with torch.no_grad():
                mu, std = teacher.dist_params(states)
            m_mu, m_std = self.meta_actor.dist_params(states)
            return torch.mean(torch.square(m_mu - mu).sum(-1) + torch.square(m_std - std).sum(-1))

        for epoch in range(cfg["epoch_meta"]):
            for _ in range(steps):
                loss = torch.zeros((), device=self.device)
                if new_ready:  # new knowledge: match the fast learner on this task's states
                    loss = loss + wasserstein(self.fast2meta.sample(batch_size), self.fast_actor)
                if previous_meta is not None:  # old knowledge: match the previous meta learner on past states
                    loss = loss + wasserstein(self.meta_buffer.sample(batch_size), previous_meta)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch == 0 or (epoch + 1) % 50 == 0:
                print(f"[FAME] Meta epoch {epoch + 1}/{cfg['epoch_meta']} loss={loss.item():.3e}")
            if (epoch + 1) % 2 == 0:
                scheduler.step()
        self._meta_trained = True


class Fame(Method):
    name = "fame"
    reuses_policies = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = dict(self.method_cfg)
        cfg["rollout_steps"] = self.agent_cfg["agent"]["rollouts"]
        cfg["experiment"] = dict(self.agent_cfg["agent"]["experiment"])
        self.agent = FAMEAgent(self.env.observation_space, self.env.action_space, self.device, cfg)

    def _evaluate(self, actor: SquashedActor, num_steps: int) -> float:
        """Mean return of the episodes that end within ``num_steps`` steps of the deterministic actor."""
        self.env._reset_once = True
        obs, _ = self.env.reset()
        returns = torch.zeros(self.env.num_envs, device=self.device)
        completed = []
        with torch.no_grad():
            for _ in range(num_steps):
                obs, rewards, terminated, truncated, _ = self.env.step(actor(obs).mean)
                returns += rewards.view(-1)
                done = (terminated | truncated).view(-1)
                completed.append(returns[done])
                returns[done] = 0.0
        completed = torch.cat(completed)
        # the trainer's first reset must reset the environment again
        self.env._reset_once = True
        return completed.mean().item() if completed.numel() else 0.0

    def start_stage(self, index, task, stage_dir):
        self.agent.experiment_dir = stage_dir
        self.agent.detection = None
        self.agent.start_task(task, self._evaluate, first=index == 0)
        return self.agent

    def end_stage(self, index, task, stage_dir, agent):
        self.load_best_checkpoint(agent, stage_dir)
        if self.stage_info["best_timestep"] < self.agent_cfg["trainer"]["timesteps"]:
            agent.fast2meta.clear()
            agent.fast2meta.add(self._visited_states(agent.fast_actor, task))
            print(f"[FAME] Re-collected {agent.fast2meta.size} states of {task} with the best fast actor")
        policy = SquashedMean(copy.deepcopy(agent.fast_actor))
        agent.end_task()
        torch.save(SquashedMean(copy.deepcopy(agent.meta_actor)), os.path.join(stage_dir, "meta_policy.pt"))
        return policy

    def resume(self, stages):
        meta = self.load_module(os.path.join(os.path.dirname(stages[-1]["policy"]), "meta_policy.pt"))
        self.agent.meta_actor.load_state_dict(meta.actor.state_dict())
        self.agent._meta_trained = True
        self.agent.meta_buffer.clear()
        for stage in stages:
            states = self._visited_states(self.load_module(stage["policy"]).actor, stage["task"])
            self.agent.meta_buffer.add(states)
            print(f"[FAME] Refilled the meta buffer with {states.shape[0]} states of {stage['task']}")

    def _visited_states(self, actor: SquashedActor, task: str) -> torch.Tensor:
        """The last ``size_fast2meta`` states visited by the stochastic ``actor`` on ``task``.

        One episode length of warm-up comes first, so that the episodes are spread over their length
        as in training.
        """
        self.env.unwrapped.set_task(task)
        self.env._reset_once = True
        obs, _ = self.env.reset()
        num_steps = math.ceil(self.agent.fast2meta.capacity / self.env.num_envs)
        states = []
        with torch.no_grad():
            for step in range(int(self.env.unwrapped.max_episode_length) + num_steps):
                if step >= int(self.env.unwrapped.max_episode_length):
                    states.append(obs.clone())
                obs, _, _, _, _ = self.env.step(actor(obs).sample())
        self.env._reset_once = True
        return torch.cat(states)[-self.agent.fast2meta.capacity :]
