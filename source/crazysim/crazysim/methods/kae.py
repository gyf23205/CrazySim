"""Koopman autoencoder (KAE) that approximates a policy's observation-to-action map.

Ported from ``less_leg_walking_1`` (``agents/KAE_approx.py``, ``Autoencoder.py``, ``utils.get_experts_outputs``).

The KAE encodes a padded observation ``x`` into ``z``, and a linear operator ``K`` maps ``z`` to the
encoding of the padded policy action ``y``: ``decoder(K z) ~ y``. ``K`` is refit by least squares
after every optimizer step. The eigen-decomposition of ``K`` splits ``decoder(K z)`` into one term per
eigenvalue; each term is one expert of the KAE+MoE policy.

Changes from the reference: the training samples are drawn and labelled in one batch instead of
one sample at a time, the eigen-decomposition is computed once when a :class:`KAEExperts` is built
instead of in every forward pass, and the summary also reports how well the KAE reproduces the
policy's actions on observed states.
"""

from __future__ import annotations

import json
import os
import random
import time
import torch
import torch.nn as nn
from collections.abc import Callable
from torch.distributions import MultivariateNormal


class KoopmanAutoencoder(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int, observable_dim: int, device):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim * 4),
            nn.Tanh(),
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.Tanh(),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.Tanh(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, observable_dim),
        )
        self.decoder = nn.Linear(observable_dim, state_dim, bias=False)
        self.state_dim = state_dim
        self.observable_dim = observable_dim
        self.K = torch.randn(observable_dim, observable_dim, device=device)

    def forward(self, x):
        z = self.encoder(x)
        y_hat = self.decoder(z @ self.K.T)
        x_hat = self.decoder(z)
        return x_hat, z, y_hat

    def compute_koopman_operator(self, latent_x, latent_y):
        """Least-squares fit of ``latent_y ~ K latent_x``. Kept inside autograd, as in the reference: the
        next loss also backpropagates through this fit into the encoder."""
        latent_x = latent_x.view(-1, latent_x.size(-1)).T
        latent_y = latent_y.view(-1, latent_y.size(-1)).T
        self.K = latent_y @ torch.linalg.pinv(latent_x, rcond=1e-5)


def pad(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Pad the last dimension with ones up to ``dim``."""
    return torch.cat([x, torch.ones(*x.shape[:-1], dim - x.shape[-1], device=x.device, dtype=x.dtype)], dim=-1)


class KAEExperts(nn.Module):
    """Frozen KAE whose eigen-modes act as experts.

    ``forward(padded_obs)`` returns ``(batch, num_modes, action_dim)``; the sum over modes equals the
    first ``action_dim`` entries of ``decoder(K^p z)``.
    """

    def __init__(self, kae: KoopmanAutoencoder, action_dim: int, p: int = 1):
        super().__init__()
        self.encoder = kae.encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        K = kae.K.detach()
        eigvals, eigvec_left = torch.linalg.eig(K.T)
        eigvals = eigvals.conj()
        eigvec_left = eigvec_left.conj().T  # rows: left eigenvectors
        decoder = kae.decoder.weight.detach().to(torch.complex64)
        v = decoder @ torch.linalg.inv(eigvec_left)  # (state_dim, num_modes)
        self.register_buffer("eigvals_p", eigvals**p)
        self.register_buffer("eigvec_left", eigvec_left)
        self.register_buffer("v", v[:action_dim])
        self.num_modes = K.shape[0]

    def forward(self, padded_obs: torch.Tensor) -> torch.Tensor:
        z = self.encoder(padded_obs).to(torch.complex64)
        phi = z @ self.eigvec_left.T  # (batch, num_modes)
        outputs = (self.eigvals_p * phi).unsqueeze(1) * self.v  # (batch, action_dim, num_modes)
        return outputs.transpose(1, 2).real.float()


def _set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _fit_observation_distribution(observations: torch.Tensor, device) -> MultivariateNormal:
    observations64 = observations.to(dtype=torch.float64, device="cpu")
    mu = observations64.mean(dim=0)
    centered = observations64 - mu
    sigma = centered.T @ centered / (observations64.shape[0] - 1)
    sigma = 0.5 * (sigma + sigma.T) + 1e-5 * torch.eye(sigma.shape[0], dtype=sigma.dtype)
    return MultivariateNormal(mu.float().to(device), covariance_matrix=sigma.float().to(device))


def _koopman_loss(kae, x, y_seq, p, action_dim):
    """Reconstruction, multi-step state and latent prediction losses, and the action loss."""
    mse = nn.MSELoss()
    x_hat, latent_x, _ = kae(x)
    recon_loss = mse(x_hat, x)
    state_loss = latent_loss = action_loss = 0.0
    for k in range(p):
        _, latent_yk, _ = kae(y_seq[:, k])
        pred_latent = latent_x @ torch.linalg.matrix_power(kae.K, k + 1).T
        pred_y = kae.decoder(pred_latent)
        state_loss += mse(pred_y, y_seq[:, k])
        latent_loss += mse(pred_latent, latent_yk)
        if k == p - 1:
            action_loss = mse(pred_y[:, :action_dim], y_seq[:, k, :action_dim])
    return recon_loss, state_loss / p, latent_loss / p, action_loss


def train_kae(
    policy: Callable[[torch.Tensor], torch.Tensor],
    observations: torch.Tensor,
    padded_dim: int,
    observable_dim: int,
    p: int,
    hidden_dim: int = 256,
    sample_count: int = 15000,
    batch_size: int = 2048,
    epochs: int = 7500,
    learning_rate: float = 1e-4,
    kae_coefficient: float = 0.1,
    action_coefficient: float = 0.9,
    c1: float = 1.0,
    c2: float = 1.0,
    c3: float = 1.0,
    seed: int = 777,
    log_dir: str | None = None,
) -> KoopmanAutoencoder:
    """Fit a KAE to ``policy`` on states drawn from a Gaussian fitted to ``observations``."""
    device = observations.device
    _set_seed(seed)
    obs_dim = observations.shape[-1]

    # training data: sampled observations and the policy's deterministic actions, both padded
    distribution = _fit_observation_distribution(observations, device)
    with torch.no_grad():
        inputs = distribution.sample((sample_count,))
        outputs = policy(inputs)
    action_dim = outputs.shape[-1]
    if padded_dim < max(obs_dim, action_dim):
        raise ValueError(f"padded_dim={padded_dim} is smaller than obs_dim={obs_dim} or action_dim={action_dim}.")
    inputs, outputs = pad(inputs, padded_dim), pad(outputs, padded_dim)

    kae = KoopmanAutoencoder(padded_dim, hidden_dim, observable_dim, device).to(device)
    optimizer = torch.optim.Adam(kae.parameters(), lr=learning_rate)
    history = []
    start = time.perf_counter()
    num_batches = (sample_count + batch_size - 1) // batch_size
    for epoch in range(epochs):
        sums = torch.zeros(6, device=device)
        for begin in range(0, sample_count, batch_size):
            end = min(begin + batch_size, sample_count)
            # the k-th prediction target of sample i is the output of sample i + k; past the end of the
            # data the last available target is repeated (as in the reference)
            y_seq = [outputs[begin + k : end + k] for k in range(p) if end + k <= sample_count]
            y_seq = torch.stack(y_seq + [y_seq[-1]] * (p - len(y_seq)), dim=1)
            recon, state, latent, action = _koopman_loss(kae, inputs[begin:end], y_seq, p, action_dim)
            loss_kae = c1 * recon + c2 * state + c3 * latent
            loss = kae_coefficient * loss_kae + action_coefficient * action
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite KAE loss at epoch {epoch + 1}.")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(kae.parameters(), max_norm=1.0)
            optimizer.step()
            sums += torch.stack([loss, loss_kae, action, recon, state, latent]).detach()

            _, latent_in, _ = kae(inputs)
            _, latent_out, _ = kae(outputs)
            kae.compute_koopman_operator(latent_in, latent_out)
            if not torch.isfinite(kae.K).all():
                raise FloatingPointError(f"Koopman operator contains NaN or Inf at epoch {epoch + 1}.")

        if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == epochs:
            means = (sums / num_batches).tolist()
            history.append([epoch + 1, *means, time.perf_counter() - start])
            print(
                f"[KAE] epoch {epoch + 1}/{epochs} loss={means[0]:.6f} action={means[2]:.6f} "
                f"recon={means[3]:.6f} state={means[4]:.6f} latent={means[5]:.6f}",
                flush=True,
            )

    kae.K = kae.K.detach()
    kae.eval()
    for param in kae.parameters():
        param.requires_grad = False

    # quality on the observed states (not the sampled ones)
    with torch.no_grad():
        real = observations[:4096]
        real_pad = pad(real, padded_dim)
        real_hat, _, action_hat = kae(real_pad)
        actions = policy(real)
        summary = {
            "epochs": epochs,
            "sample_count": sample_count,
            "padded_dim": padded_dim,
            "observable_dim": observable_dim,
            "final_loss": history[-1][1],
            "final_action_loss": history[-1][3],
            "eval_recon_relative_error": (torch.norm(real_hat - real_pad) / torch.norm(real_pad)).item(),
            "eval_action_mse": nn.functional.mse_loss(action_hat[:, :action_dim], actions).item(),
            "eval_action_variance": actions.var(dim=0).mean().item(),
            "spectral_radius": torch.linalg.eigvals(kae.K).abs().max().item(),
            "elapsed_seconds": time.perf_counter() - start,
        }
    print(
        f"[KAE] action MSE on observed states {summary['eval_action_mse']:.5f} "
        f"(action variance {summary['eval_action_variance']:.5f}), {summary['elapsed_seconds']:.0f} s"
    )
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "kae_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(log_dir, "kae_history.csv"), "w") as f:
            f.write("epoch,loss,loss_kae,action,recon,state,latent,elapsed_s\n")
            f.writelines(",".join(str(v) for v in row) + "\n" for row in history)
    return kae
