"""Transfer methods for training one policy per task over a task sequence (see :mod:`.base`)."""

from .base import FINAL_CHECKPOINT, POLICY_FILE, DeterministicPolicy, Method
from .componet import Componet
from .fame import Fame
from .kae_moe import KaeMoe
from .ppo import Finetune, Scratch
from .residual import Residual

METHODS: dict[str, type[Method]] = {m.name: m for m in [Finetune, Scratch, Residual, Componet, KaeMoe, Fame]}
