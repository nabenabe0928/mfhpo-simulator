from __future__ import annotations

import os
from typing import ClassVar, Final

import ConfigSpace as CS

from benchmark_apis.hpo.abstract_bench import AbstractBench, DATA_DIR_NAME

from yahpo_gym import benchmark_set, local_config


FIDEL_KEY = "epoch"


class LCBenchSurrogate:
    """Workaround to prevent dask from serializing the objective func"""

    def __init__(self, dataset_id: str, target_metric: str):
        benchdata_path = os.path.join(DATA_DIR_NAME, "lcbench")
        self._check_benchdata_availability(benchdata_path)
        self._target_metric = target_metric
        self._dataset_id = dataset_id
        # active_session=False is necessary for parallel computing.
        self._surrogate = benchmark_set.BenchmarkSet("lcbench", instance=dataset_id, active_session=False)

    def _check_benchdata_availability(self, benchdata_path: str) -> None:
        if not os.path.exists(benchdata_path):
            raise FileNotFoundError(
                f"Could not find the dataset at {benchdata_path}.\n"
                f"Download the dataset and place the file at {benchdata_path}.\n"
                "You can download the dataset via:\n"
                "\t$ wget https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/\n\n"
                f"Then unzip the file in {DATA_DIR_NAME}."
            )

        local_config.init_config()
        local_config.set_data_path(DATA_DIR_NAME)

    def __call__(self, eval_config: dict[str, int | float], fidel: int) -> dict[str, float]:
        _eval_config = eval_config.copy()
        _eval_config["OpenML_task_id"] = self._dataset_id
        _eval_config[FIDEL_KEY] = fidel
        output = self._surrogate.objective_function(_eval_config)[0]
        return dict(loss=float(1.0 - output[self._target_metric]), runtime=float(output["time"]))


class LCBench(AbstractBench):
    # https://syncandshare.lrz.de/getlink/fiCMkzqj1bv1LfCUyvZKmLvd/
    _target_metric: ClassVar[str] = "val_balanced_accuracy"
    _TRUE_MAX_FIDEL: Final[ClassVar[int]] = 52
    _N_DATASETS: Final[ClassVar[int]] = 34
    _DATASET_NAMES: Final[tuple[str]] = (
        "kddcup09",
        "covertype",
        "amazon-employee-access",
        "adult",
        "nomao",
        "bank-marketing",
        "shuttle",
        "australian",
        "kr-vs-kp",
        "mfeat-factors",
        "credit-g",
        "vehicle",
        "kc1",
        "blood-transfusion-service-center",
        "cnae-9",
        "phoneme",
        "higgs",
        "connect-4",
        "helena",
        "jannis",
        "volkert",
        "mini-boo-ne",
        "aps-failure",
        "christine",
        "fabert",
        "airlines",
        "jasmine",
        "sylvine",
        "albert",
        "dionis",
        "car",
        "segment",
        "fashion-mnist",
        "jungle-chess-2pcs-raw-endgame-complete",
    )

    def __init__(
        self,
        dataset_id: int,
        seed: int | None = None,  # surrogate is not stochastic
        keep_benchdata: bool = True,
    ):
        dataset_info = (
            ("kddcup09_appetency", "3945"),
            ("covertype", "7593"),
            ("amazon_employee_access", "34539"),
            ("adult", "126025"),
            ("nomao", "126026"),
            ("bank_marketing", "126029"),
            ("shuttle", "146212"),
            ("australian", "167104"),
            ("kr_vs_kp", "167149"),
            ("mfeat_factors", "167152"),
            ("credit_g", "167161"),
            ("vehicle", "167168"),
            ("kc1", "167181"),
            ("blood_transfusion_service_center", "167184"),
            ("cnae_9", "167185"),
            ("phoneme", "167190"),
            ("higgs", "167200"),
            ("connect_4", "167201"),
            ("helena", "168329"),
            ("jannis", "168330"),
            ("volkert", "168331"),
            ("mini_boo_ne", "168335"),
            ("aps_failure", "168868"),
            ("christine", "168908"),
            ("fabert", "168910"),
            ("airlines", "189354"),
            ("jasmine", "189862"),
            ("sylvine", "189865"),
            ("albert", "189866"),
            ("dionis", "189873"),
            ("car", "189905"),
            ("segment", "189906"),
            ("fashion_mnist", "189908"),
            ("jungle_chess_2pcs_raw_endgame_complete", "189909"),
        )
        self.dataset_name, self._dataset_id = dataset_info[dataset_id]
        self._surrogate = self.get_benchdata() if keep_benchdata else None
        self._config_space = self.config_space

    def get_benchdata(self) -> LCBenchSurrogate:
        return LCBenchSurrogate(dataset_id=self._dataset_id, target_metric=self._target_metric)

    def _validate_config(self, eval_config: dict[str, int | float]) -> None:
        EPS = 1e-12
        for hp in self._config_space.get_hyperparameters():
            lb, ub, name = hp.lower, hp.upper, hp.name
            if isinstance(hp, CS.UniformFloatHyperparameter):
                assert isinstance(eval_config[name], float) and lb - EPS <= eval_config[name] <= ub + EPS
            else:
                eval_config[name] = int(eval_config[name])
                assert isinstance(eval_config[name], int) and lb <= eval_config[name] <= ub

    def __call__(
        self,
        eval_config: dict[str, int | float],
        *,
        fidels: dict[str, int | float] = {FIDEL_KEY: 52},
        seed: int | None = None,
        benchdata: LCBenchSurrogate | None = None,
    ) -> dict[str, float]:
        if benchdata is None and self._surrogate is None:
            raise ValueError("data must be provided when `keep_benchdata` is False")

        surrogate = benchdata if self._surrogate is None else self._surrogate
        fidel = int(min(self._TRUE_MAX_FIDEL, fidels[FIDEL_KEY]))
        self._validate_config(eval_config=eval_config)
        return surrogate(eval_config=eval_config, fidel=fidel)

    @property
    def config_space(self) -> CS.ConfigurationSpace:
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameters(
            [
                CS.UniformIntegerHyperparameter(name="batch_size", lower=16, upper=512, log=True),
                CS.UniformFloatHyperparameter(name="learning_rate", lower=1e-4, upper=0.1, log=True),
                CS.UniformFloatHyperparameter(name="max_dropout", lower=0.0, upper=1.0),
                CS.UniformIntegerHyperparameter(name="max_units", lower=64, upper=1024, log=True),
                CS.UniformFloatHyperparameter(name="momentum", lower=0.1, upper=0.9),
                CS.UniformIntegerHyperparameter(name="num_layers", lower=1, upper=5),
                CS.UniformFloatHyperparameter(name="weight_decay", lower=1e-5, upper=0.1),
            ]
        )
        return config_space

    @property
    def min_fidels(self) -> dict[str, int | float]:
        return {FIDEL_KEY: 6}

    @property
    def max_fidels(self) -> dict[str, int | float]:
        # in reality, the max_fidel is 52, but make it 54 only for computational convenience.
        return {FIDEL_KEY: 54}

    @property
    def fidel_keys(self) -> list[str]:
        return [FIDEL_KEY]
