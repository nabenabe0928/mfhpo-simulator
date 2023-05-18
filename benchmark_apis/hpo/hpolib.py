import json
import os
import pickle
from typing import Dict, List, Optional, TypedDict, Union

import ConfigSpace as CS

import numpy as np

from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, VALUE_RANGES


class RowDataType(TypedDict):
    valid_mse: List[Dict[int, float]]
    runtime: List[float]


class HPOLibDatabase:
    """Workaround to prevent dask from serializing the objective func"""

    def __init__(self, dataset_name: str):
        benchdata_path = os.path.join(DATA_DIR_NAME, "hpolib", f"{dataset_name}.pkl")
        self._check_benchdata_availability(benchdata_path)
        self._db = pickle.load(open(benchdata_path, "rb"))

    def _check_benchdata_availability(self, benchdata_path: str) -> None:
        if not os.path.exists(benchdata_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {benchdata_path}.\n"
                f"Download the dataset and place the file at {benchdata_path}.\n"
                "You can download the dataset via:\n"
                "\t$ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz\n"
                "\t$ tar xf fcnet_tabular_benchmarks.tar.gz\n\n"
                "Then extract the pkl file using https://github.com/nabenabe0928/hpolib-extractor."
            )

    def __getitem__(self, key: str) -> Dict[str, RowDataType]:
        return self._db[key]


class HPOLib(AbstractBench):
    """
    Download the datasets via:
        $ wget http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
        $ tar xf fcnet_tabular_benchmarks.tar.gz

    Use https://github.com/nabenabe0928/hpolib-extractor to extract the pickle file.
    """

    _N_DATASETS = 4
    _DATASET_NAMES = ("slice-localization", "protein-structure", "naval-propulsion", "parkinsons-telemonitoring")

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int],
        keep_benchdata: bool = True,
    ):
        self.dataset_name = [
            "slice_localization",
            "protein_structure",
            "naval_propulsion",
            "parkinsons_telemonitoring",
        ][dataset_id]
        self._db = self.get_benchdata() if keep_benchdata else None
        self._rng = np.random.RandomState(seed)
        self._value_range = VALUE_RANGES["hpolib"]

    def get_benchdata(self) -> HPOLibDatabase:
        return HPOLibDatabase(self.dataset_name)

    def __call__(
        self,
        eval_config: Dict[str, Union[int, str]],
        fidel: int = 100,
        seed: Optional[int] = None,
        benchdata: Optional[HPOLibDatabase] = None,
    ) -> Dict[str, float]:
        if benchdata is None and self._db is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        db = benchdata if self._db is None else self._db
        fidel = int(fidel)
        idx = seed % 4 if seed is not None else self._rng.randint(4)
        key = json.dumps({k: self._value_range[k][int(v)] for k, v in eval_config.items()}, sort_keys=True)
        loss = db[key]["valid_mse"][idx][fidel - 1]
        runtime = db[key]["runtime"][idx] * fidel / self.max_fidel
        return dict(loss=np.log(loss), runtime=runtime)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        return self._fetch_discrete_config_space()

    @property
    def min_fidel(self) -> int:
        return 11

    @property
    def max_fidel(self) -> int:
        return 100
