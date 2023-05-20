from argparse import ArgumentParser, Namespace
from typing import Any

from benchmark_apis.hpo.hpolib import HPOLib
from benchmark_apis.hpo.jahs import JAHSBench201
from benchmark_apis.hpo.lcbench import LCBench
from benchmark_apis.synthetic.branin import MFBranin
from benchmark_apis.synthetic.hartmann import MFHartmann


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201, branin=MFBranin, hartmann=MFHartmann)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_id", type=int, default=0, choices=list(range(34)))
    parser.add_argument("--dim", type=int, default=3, choices=[3, 6], help="Only for Hartmann")
    parser.add_argument("--bench_name", type=str, choices=list(BENCH_CHOICES.keys()))
    parser.add_argument("--n_workers", type=int)
    args = parser.parse_args()
    return args


def get_subdir_name(args: Namespace) -> str:
    dataset_part = ""
    dataset_names = BENCH_CHOICES[args.bench_name]._DATASET_NAMES
    if dataset_names is not None:
        dataset_part = f"_dataset={dataset_names[args.dataset_id]}"

    bench_name = args.bench_name
    if args.bench_name == "hartmann":
        bench_name = f"{args.bench_name}{args.dim}d"

    return f"bench={bench_name}{dataset_part}_nworkers={args.n_workers}/{args.seed}"


def get_bench_instance(args: Namespace, keep_benchdata: bool = True) -> Any:
    bench_cls = BENCH_CHOICES[args.bench_name]
    if bench_cls._BENCH_TYPE == "HPO":
        obj_func = bench_cls(dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=keep_benchdata)
    else:
        kwargs = dict(dim=args.dim) if args.bench_name == "hartmann" else dict()
        obj_func = bench_cls(seed=args.seed, **kwargs)

    return obj_func
