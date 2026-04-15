from __future__ import annotations

from typing import Any

import numpy as np


def _two_dicts_almost_equal(d1: dict[str, Any], d2: dict[str, Any]) -> bool:
    """for atol and rtol, I referred to numpy.isclose"""
    if set(d1.keys()) != set(d2.keys()):
        return False

    for k in d1.keys():
        v1, v2 = d1[k], d2[k]
        if isinstance(v1, (float, int)) and isinstance(v2, (float, int)):
            if not np.isclose(v1, v2):
                return False
        elif v1 != v2:
            return False

    return True


class _AskTellConfigIDTracker:
    def __init__(self) -> None:
        self._existing_configs: dict[str, dict[str, Any]] = {}

    def validate(self, config: dict[str, Any], config_id: int) -> None:
        config_id_str = str(config_id)
        if config_id_str not in self._existing_configs:
            self._existing_configs[config_id_str] = config.copy()
            return

        existing_config = self._existing_configs[config_id_str]
        if not _two_dicts_almost_equal(existing_config, config):
            raise ValueError(
                f"{config_id=} already exists ({existing_config=}), but got the duplicated config_id for {config=}"
            )
