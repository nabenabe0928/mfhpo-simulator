import os
from typing import Dict, Optional, Union

import ConfigSpace as CS

from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME, VALUE_RANGES

try:
    import jahs_bench
except ModuleNotFoundError:  # We cannot use jahs with smac
    pass


class JAHSBenchSurrogate:
    """Workaround to prevent dask from serializing the objective func"""

    def __init__(self, data_dir: str, dataset_name: str, target_metric: str):
        self._check_benchdata_availability(benchdata_path=data_dir)
        self._target_metric = target_metric
        self._surrogate = jahs_bench.Benchmark(
            task=dataset_name, download=False, save_dir=data_dir, metrics=[self._target_metric, "runtime"]
        )

    def _check_benchdata_availability(self, benchdata_path: str) -> None:
        data_url = (
            "https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar"
        )
        if not os.path.exists(benchdata_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {benchdata_path}.\n"
                f"Download the dataset and place the file at {benchdata_path}.\n"
                "You can download the dataset via:\n"
                f"\t$ wget {data_url}\n\n"
                f"Then untar the file in {benchdata_path}."
            )

    def __call__(self, eval_config: Dict[str, Union[int, str, float]], fidel: int = 200) -> Dict[str, float]:
        eval_config.update(Optimizer="SGD", Resolution=1.0)
        eval_config = {k: int(v) if k[:-1] == "Op" else v for k, v in eval_config.items()}
        output = self._surrogate(eval_config, nepochs=fidel)[fidel]
        return dict(loss=100 - output[self._target_metric], runtime=output["runtime"])


class JAHSBench201(AbstractBench):
    # https://ml.informatik.uni-freiburg.de/research-artifacts/jahs_bench_201/v1.1.0/assembled_surrogates.tar
    _target_metric = "valid-acc"
    _N_DATASETS = 3
    _DATASET_NAMES = ("cifar10", "fashion-mnist", "colorectal-histology")

    def __init__(
        self,
        dataset_id: int,
        seed: Optional[int] = None,  # surrogate is not stochastic
        keep_benchdata: bool = True,
    ):
        self.dataset_name = ["cifar10", "fashion_mnist", "colorectal_histology"][dataset_id]
        self._data_dir = os.path.join(DATA_DIR_NAME, "jahs_bench_data")
        self._surrogate = self.get_benchdata() if keep_benchdata else None
        self._value_range = VALUE_RANGES["jahs-bench"]

    def get_benchdata(self) -> JAHSBenchSurrogate:
        return JAHSBenchSurrogate(
            data_dir=self._data_dir, dataset_name=self.dataset_name, target_metric=self._target_metric
        )

    def __call__(
        self,
        eval_config: Dict[str, Union[int, str, float]],
        fidel: int = 200,
        seed: Optional[int] = None,
        benchdata: Optional[JAHSBenchSurrogate] = None,
    ) -> Dict[str, float]:
        if benchdata is None and self._surrogate is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        surrogate = benchdata if self._surrogate is None else self._surrogate
        fidel = int(fidel)
        EPS = 1e-12
        _eval_config = {
            k: self._value_range[k][int(v)] if k in self._value_range else float(v) for k, v in eval_config.items()
        }
        assert isinstance(_eval_config["LearningRate"], float)
        assert 1e-3 - EPS <= _eval_config["LearningRate"] <= 1.0 + EPS
        assert isinstance(_eval_config["WeightDecay"], float)
        assert 1e-5 - EPS <= _eval_config["WeightDecay"] <= 1e-2 + EPS
        return surrogate(_eval_config, fidel)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = self._fetch_discrete_config_space()
        config_space.add_hyperparameters(
            [
                CS.UniformFloatHyperparameter(name="LearningRate", lower=1e-3, upper=1.0, log=True),
                CS.UniformFloatHyperparameter(name="WeightDecay", lower=1e-5, upper=1e-2, log=True),
            ]
        )
        return config_space

    @property
    def min_fidel(self) -> int:
        return 22

    @property
    def max_fidel(self) -> int:
        return 200