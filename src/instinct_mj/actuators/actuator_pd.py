from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass
    # NOTE: RandomizePDActuator is duplicated with mdp.randomize_actuator_gains event.
