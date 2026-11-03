"""Export a trained policy and its KAE in the artifact formats of ``Drone_Test_TRO``.

Needs only torch, not Isaac Sim. Writes ``<stage>/deploy/``:

    deployable_policy_checkpoint.pt   for run.py --method eval --policy-path
    kae.pth                           for run.py --method kae --kae-checkpoint and --method kae_moe --kae-checkpoints

The KAE is pickled as ``Autoencoder.KoopmanAutoencoder``, the class in ``examples/setpoint/kae_models``, which is what
that repository's loader finds automatically.

Example:
    python scripts/skrl/export_waypoint_artifacts.py --stage_dir logs/skrl/crazyflie_tracking_waypoint/<session>/0_square
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage_dir", type=str, required=True)
parser.add_argument("--drone_test_dir", type=str, default=os.path.expanduser("~/git/Drone_Test_TRO"))
parser.add_argument("--setpoint_count", type=int, default=16)
parser.add_argument("--waypoint_offset_limit", type=float, nargs=3, default=[0.5, 0.5, 0.25])
args = parser.parse_args()

limits = ",".join(f"{value:g}" for value in args.waypoint_offset_limit)
CONTRACT = f"trajectory_setpoints_relative_waypoint_v1:setpoints={args.setpoint_count}:offset_limit={limits}"
"""``observation_contract`` of ``examples/trajectory_setpoints/observation.py``."""
ACTOR_KEYS = {"actor.0": "net_container.0", "actor.2": "net_container.2", "actor.4": "policy_layer.0"}
"""Layers of the actor here and of ``BoundedScratchPolicy`` there."""


def rename_policy_state(state: dict) -> OrderedDict:
    # the loader of Drone_Test_TRO accepts state dictionaries only as OrderedDict
    renamed = OrderedDict(log_std_parameter=state["log_std_parameter"].cpu())
    for old, new in ACTOR_KEYS.items():
        for name in ("weight", "bias"):
            renamed[f"{new}.{name}"] = state[f"{old}.{name}"].cpu()
    return renamed


deploy_dir = os.path.join(args.stage_dir, "deploy")
os.makedirs(deploy_dir, exist_ok=True)

checkpoint = torch.load(os.path.join(args.stage_dir, "checkpoints", "best_agent.pt"), map_location="cpu", weights_only=False)
policy_state = rename_policy_state(checkpoint["policy"])
torch.save(
    {
        "policy": policy_state,
        "state_preprocessor": OrderedDict((k, v.cpu()) for k, v in checkpoint["state_preprocessor"].items()),
        "observation_contract": CONTRACT,
    },
    os.path.join(deploy_dir, "deployable_policy_checkpoint.pt"),
)
print(f"[INFO] Wrote {os.path.join(deploy_dir, 'deployable_policy_checkpoint.pt')}")

kae_path = os.path.join(args.stage_dir, "kae", "kae_state.pt")
if os.path.isfile(kae_path):
    sys.path.insert(0, os.path.join(args.drone_test_dir, "examples", "setpoint", "kae_models"))
    from Autoencoder import KoopmanAutoencoder

    saved = torch.load(kae_path, map_location="cpu", weights_only=True)
    kae = KoopmanAutoencoder(saved["state_dim"], saved["hidden_dim"], saved["observable_dim"], "cpu")
    kae.encoder.encoder.load_state_dict(saved["encoder"])
    kae.decoder.linear.load_state_dict(saved["decoder"])
    kae.K = saved["K"].cpu()
    torch.save(kae, os.path.join(deploy_dir, "kae.pth"))
    print(f"[INFO] Wrote {os.path.join(deploy_dir, 'kae.pth')}")
else:
    print(f"[INFO] No KAE at {kae_path}; run scripts/skrl/fit_kae.py first")
