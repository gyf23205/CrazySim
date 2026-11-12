"""Hardware-free checks of the warm-up handoff (docs/warm_start.md).

They run the method classes of this repository on the CPU with a small tensor environment, and read the
written bundles with the loader of the adjacent ``Drone_Test_TRO`` checkout (``DRONE_TEST_ROOT`` when it is
elsewhere). They do not start Isaac Sim and say nothing about tracking quality or flight.
"""

import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRONE_TEST_ROOT = Path(os.environ.get("DRONE_TEST_ROOT", ROOT.parent / "Drone_Test_TRO"))

if "crazysim" not in sys.modules:
    # ``import crazysim`` registers the environments, which needs Isaac Sim. The methods only need the
    # type of the skrl environment wrapper for an annotation.
    package = types.ModuleType("crazysim")
    package.__path__ = [str(ROOT / "source" / "crazysim" / "crazysim")]
    sys.modules["crazysim"] = package
    wrapper = types.ModuleType("isaaclab_rl.skrl")
    wrapper.SkrlVecEnvWrapper = object
    sys.modules.setdefault("isaaclab_rl", types.ModuleType("isaaclab_rl"))
    sys.modules.setdefault("isaaclab_rl.skrl", wrapper)
