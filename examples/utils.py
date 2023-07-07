from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any

from benchmark_apis import HPOLib, JAHSBench201, LCBench, MFBranin, MFHartmann


BENCH_CHOICES = dict(lc=LCBench, hpolib=HPOLib, jahs=JAHSBench201, branin=MFBranin, hartmann=MFHartmann)


@dataclass(frozen=True)
class ParsedArgs:
    seed: int
    dataset_id: int
    dim: int
    bench_name: str
    n_workers: int
    worker_index: int | None


def parse_args() -> ParsedArgs:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset_id", type=int, default=0, choices=list(range(34)))
    parser.add_argument("--dim", type=int, default=3, choices=[3, 6], help="Only for Hartmann")
    parser.add_argument("--bench_name", type=str, choices=list(BENCH_CHOICES.keys()))
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--worker_index", type=int, default=None)
    args = parser.parse_args()

    kwargs = {k: getattr(args, k) for k in ParsedArgs.__annotations__.keys()}
    return ParsedArgs(**kwargs)


def get_save_dir_name(args: ParsedArgs) -> str:
    dataset_part = ""
    if BENCH_CHOICES[args.bench_name]._BENCH_TYPE == "HPO":
        dataset_name = "-".join(BENCH_CHOICES[args.bench_name]._CONSTS.dataset_names[args.dataset_id].split("_"))
        dataset_part = f"_dataset={dataset_name}"

    bench_name = args.bench_name
    if args.bench_name == "hartmann":
        bench_name = f"{args.bench_name}{args.dim}d"

    return f"bench={bench_name}{dataset_part}_nworkers={args.n_workers}/{args.seed}"


def get_bench_instance(args: ParsedArgs, keep_benchdata: bool = True) -> Any:
    bench_cls = BENCH_CHOICES[args.bench_name]
    if bench_cls._BENCH_TYPE == "HPO":
        obj_func = bench_cls(dataset_id=args.dataset_id, seed=args.seed, keep_benchdata=keep_benchdata)
    else:
        kwargs = dict(dim=args.dim) if args.bench_name == "hartmann" else dict()
        obj_func = bench_cls(seed=args.seed, **kwargs)

    return obj_func
