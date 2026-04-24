from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark_simulator._constants import AbstractAskTellOptimizer
from benchmark_simulator.simulator import get_multiple_wrappers
from benchmark_simulator.simulator import ObjectiveFuncWrapper


if TYPE_CHECKING:
    from benchmark_simulator._constants import ObjectiveFuncType


__version__ = "1.5.0"
__copyright__ = "Copyright (C) 2026 Shuhei Watanabe"
__licence__ = "Apache-2.0 License"
__author__ = "Shuhei Watanabe"
__author_email__ = "shuhei.watanabe.utokyo@gmail.com"
__url__ = "https://github.com/nabenabe0928/mfhpo-simulator"


__all__ = [
    "AbstractAskTellOptimizer",
    "ObjectiveFuncWrapper",
    "ObjectiveFuncType",
    "get_multiple_wrappers",
]
