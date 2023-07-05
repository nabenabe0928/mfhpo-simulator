from __future__ import annotations

from typing import Any

from benchmark_simulator._secure_proc import _fetch_existing_configs, _record_existing_configs
from benchmark_simulator._utils import _SecureLock

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


class _ConfigIDTracker:
    def __init__(self, path: str, lock: _SecureLock):
        self._path = path
        self._lock = lock

    def _fetch_existing_configs(self) -> dict[str, dict[str, Any]]:
        return _fetch_existing_configs(path=self._path, lock=self._lock)

    def _record_existing_configs(self, config_id_str: str, config: dict[str, Any]) -> None:
        _record_existing_configs(path=self._path, config_id_str=config_id_str, config=config, lock=self._lock)

    def validate(self, config: dict[str, Any], config_id: int) -> None:
        config_id_str = str(config_id)
        existing_configs = self._fetch_existing_configs()
        if config_id_str not in existing_configs:
            self._record_existing_configs(config_id_str=config_id_str, config=config)
            return

        existing_config = existing_configs[config_id_str]
        if not _two_dicts_almost_equal(existing_config, config):
            raise ValueError(
                f"{config_id=} already exists ({existing_config=}), but got the duplicated config_id for {config=}"
            )


class _AskTellConfigIDTracker(_ConfigIDTracker):
    def __init__(self) -> None:
        self._existing_configs: dict[str, dict[str, Any]] = {}

    def _fetch_existing_configs(self) -> dict[str, dict[str, Any]]:
        return self._existing_configs

    def _record_existing_configs(self, config_id_str: str, config: dict[str, Any]) -> None:
        self._existing_configs[config_id_str] = config.copy()
