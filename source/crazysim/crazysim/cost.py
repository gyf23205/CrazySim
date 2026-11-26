"""Cost of a policy: memory and inference compute (Díaz-Rodríguez et al., 2018, arXiv:1810.13166).

Only ``torch`` is imported here, so the module also works outside Isaac Sim (``scripts/skrl/continual_metrics.py``).
"""

from __future__ import annotations

import statistics
import time
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode


def policy_cost(policy: nn.Module, obs_dim: int) -> dict:
    """Memory and compute of one forward pass of ``policy`` on one observation.

    Returns:
        ``parameters``: number of parameters, frozen ones included.
        ``trainable_parameters``: number of parameters trained in the policy's stage (``requires_grad``; the
        saved policies keep the flag, so modules of earlier stages that the method froze are left out).
        ``bytes``: memory of the parameters and buffers (such as fixed eigenvectors).
        ``flops_per_action``: floating-point operations of the matrix products of one forward pass
        (one multiply-add counts as 2), from :class:`torch.utils.flop_counter.FlopCounterMode`.
        Element-wise operations (activations, ``tanh``) are not counted.
    """
    tensors = list(policy.parameters()) + list(policy.buffers())
    device = tensors[0].device if tensors else torch.device("cpu")
    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        policy(torch.zeros(1, obs_dim, device=device))
    return {
        "parameters": sum(p.numel() for p in policy.parameters()),
        "trainable_parameters": sum(p.numel() for p in policy.parameters() if p.requires_grad),
        "bytes": sum(t.numel() * t.element_size() for t in tensors),
        "flops_per_action": int(counter.get_total_flops()),
    }


def cpu_latency_us(policy: nn.Module, obs_dim: int, repeats: int = 2000) -> float:
    """Median time [µs] of one forward pass on one observation, on the CPU with one thread.

    The policy is moved to the CPU. The thread count is restored afterwards.
    """
    policy = policy.cpu()
    obs = torch.zeros(1, obs_dim)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        times = []
        with torch.no_grad():
            for _ in range(100):
                policy(obs)
            for _ in range(repeats):
                start = time.perf_counter()
                policy(obs)
                times.append(time.perf_counter() - start)
    finally:
        torch.set_num_threads(threads)
    return statistics.median(times) * 1e6
