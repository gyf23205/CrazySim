"""Tanh-squashed Gaussian policy for skrl PPO.

Copy of ``SquashedNormal`` and ``SquashedGaussianMixin`` in ``examples/setpoint/expanding_kae_moe.py`` of
``Drone_Test_TRO``. The mean network outputs the pre-tanh mean ``u``; PPO samples ``tanh(z)`` with
``z ~ Normal(u, std)`` and the deterministic action is ``tanh(u)``. No clamp lies between a trainable
parameter and the action, so the policy gradient is never exactly zero.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.models.torch import Model

PRIOR_ATANH_EPSILON = 1e-3
"""Margin of :func:`prior_to_pre_tanh`. It caps ``|atanh|`` at about 3.8, so one saturated frozen policy
cannot dominate a pre-tanh sum."""


def prior_to_pre_tanh(action: torch.Tensor, epsilon: float = PRIOR_ATANH_EPSILON) -> torch.Tensor:
    """Pre-tanh value of a frozen policy's bounded action.

    Use it only on outputs of frozen modules: its clamp would zero the gradient of a trainable one.
    """
    return torch.atanh(action.clamp(-1.0 + epsilon, 1.0 - epsilon))


class SquashedNormal:
    """Tanh-squashed diagonal Normal: ``action = tanh(z)``, ``z ~ Normal(u, std)``.

    It takes the pre-tanh mean ``u`` directly. Building it from ``tanh(u)`` would need ``atanh`` of a
    clamped value, whose gradient is zero once ``tanh(u)`` rounds to the clamp bound.
    """

    def __init__(self, pre_tanh_mean: torch.Tensor, log_std: torch.Tensor, *, epsilon: float = 1e-6):
        # epsilon only guards atanh of taken actions (data) in log_prob
        self.epsilon = float(epsilon)
        self.pre_tanh_loc = pre_tanh_mean
        self.bounded_mean = torch.tanh(pre_tanh_mean)
        self.scale = log_std.exp().expand_as(pre_tanh_mean)
        self.base_dist = torch.distributions.Normal(self.pre_tanh_loc, self.scale)
        self.has_rsample = True

    @property
    def mean(self) -> torch.Tensor:
        return self.bounded_mean

    @property
    def mode(self) -> torch.Tensor:
        return self.bounded_mean

    @property
    def stddev(self) -> torch.Tensor:
        # delta-method value used only for PPO diagnostics
        return (1.0 - self.bounded_mean.square()) * self.scale

    @property
    def variance(self) -> torch.Tensor:
        return self.stddev.square()

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        return torch.tanh(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        with torch.no_grad():
            return self.rsample(sample_shape)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        bounded = value.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)
        pre_tanh = torch.atanh(bounded)
        # log(1 - tanh(x)^2), written in a stable form
        log_abs_det_jacobian = 2.0 * (math.log(2.0) - pre_tanh - F.softplus(-2.0 * pre_tanh))
        return self.base_dist.log_prob(pre_tanh) - log_abs_det_jacobian

    def entropy(self) -> torch.Tensor:
        # first-order approximation; the PPO settings use a zero entropy-loss scale
        jacobian_at_mode = torch.log1p(-self.bounded_mean.square() + self.epsilon)
        return self.base_dist.entropy() + jacobian_at_mode


class SquashedGaussianPolicy(Model):
    """Squashed Gaussian policy with a state-independent log standard deviation.

    ``actor`` returns the pre-tanh mean ``u``; ``mean_actions`` is the deterministic action ``tanh(u)``.
    """

    def __init__(self, observation_space, action_space, device, actor: nn.Module, policy_cfg: dict):
        Model.__init__(self, observation_space, action_space, device)
        self.actor = actor
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(policy_cfg["initial_log_std"])))
        self._min_log_std = float(policy_cfg.get("min_log_std", -20.0))
        self._distribution: SquashedNormal | None = None
        self._num_samples = 0

    def act(self, inputs, role=""):
        pre_tanh_mean = self.actor(inputs["states"])
        distribution = SquashedNormal(pre_tanh_mean, self.log_std_parameter.clamp(min=self._min_log_std))
        sampled_actions = distribution.rsample()
        likelihood_actions = inputs.get("taken_actions", sampled_actions)
        log_prob = distribution.log_prob(likelihood_actions).sum(dim=-1, keepdim=True)
        self._distribution = distribution
        self._num_samples = pre_tanh_mean.shape[0]
        return sampled_actions, log_prob, {"mean_actions": distribution.mean}

    def get_entropy(self, role=""):
        return self._distribution.entropy()

    def get_log_std(self, role=""):
        return self.log_std_parameter.clamp(min=self._min_log_std).repeat(self._num_samples, 1)

    def distribution(self, role=""):
        return self._distribution
