"""Pure-torch tests use unchanged methods without requiring an Isaac install."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import types

REPO_ROOT = Path(os.environ.get("CRAZYSIM_ROOT", Path(__file__).resolve().parents[1]))
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "skrl"
for entry in (SCRIPTS, REPO_ROOT / "scripts" / "skrl"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

if "isaacsim" not in sys.modules:
    # Stub only the type annotation imported by methods/base.py. Native model,
    # method, PPO and FAME implementations are loaded from their actual files.
    package = types.ModuleType("isaaclab_rl")
    package.__path__ = []
    skrl_adapter = types.ModuleType("isaaclab_rl.skrl")
    skrl_adapter.SkrlVecEnvWrapper = type("SkrlVecEnvWrapper", (), {})
    sys.modules.setdefault("isaaclab_rl", package)
    sys.modules.setdefault("isaaclab_rl.skrl", skrl_adapter)
    native = types.ModuleType("crazysim")
    native.__path__ = [str(REPO_ROOT / "source" / "crazysim" / "crazysim")]
    sys.modules.setdefault("crazysim", native)
