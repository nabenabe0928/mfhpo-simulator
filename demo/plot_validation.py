from __future__ import annotations

import json
from argparse import ArgumentParser

from benchmark_simulator.utils import get_mean_and_standard_error, get_performance_over_time

import matplotlib.pyplot as plt

import numpy as np


parser = ArgumentParser()
parser.add_argument("--mode", choices=["random", "optuna"], default="random")
args = parser.parse_args()

MODE = "-optuna" if args.mode == "optuna" else ""
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font
YMIN, YMAX = -2.3, 0.0
DataDType = dict[str, dict[str, list[list[float]]]]


def get_tmins_and_tmaxs_and_yrange(data: DataDType) -> tuple[dict[str, float], dict[str, float], float, float]:
    tmins, tmaxs = {}, {}
    for k, v in data.items():
        tmins[k] = np.min(v["actual_cumtime"])
        tmaxs[k] = np.max(v["actual_cumtime"])

    ymin, ymax = np.inf, -np.inf
    for suffix in ["deterministic", "noisy"]:
        with open(f"demo/validation{MODE}-results-{suffix}.json", mode="r") as f:
            _data = json.load(f)
            for k, v in _data.items():
                m, s = get_mean_and_standard_error(np.minimum.accumulate(v["loss"]))
                ymin = min(ymin, np.min(m - s))
                ymax = max(ymax, np.max(m + s))

    return tmins, tmaxs, ymin, ymax


def add_arrow(
    ax: plt.Axes,
    src_pos: tuple[float, float],
    target_pos: tuple[float, float],
    color: str,
    linestyle: str | None = None,
):
    ax.annotate(
        "",
        xy=target_pos,
        xytext=src_pos,
        arrowprops=dict(
            shrink=0,
            width=1,
            headwidth=8,
            headlength=10,
            connectionstyle="arc3",
            facecolor=color,
            edgecolor=color,
            linestyle=linestyle,
        ),
    )


def plot_traj(
    ax: plt.Axes,
    time_step: np.ndarray,
    traj: np.ndarray,
    color: str,
    label: str,
    linestyle: str | None = None,
    marker: str | None = None,
):
    m, s = get_mean_and_standard_error(traj)
    line = ax.plot(
        time_step,
        m,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=10,
        markersize=10,
        label=label,
        alpha=0.5,
    )[0]
    ax.fill_between(time_step, m - s, m + s, color=color, alpha=0.2)
    return line


def get_data(suffix: str):
    with open(f"demo/validation{MODE}-results-{suffix}.json", mode="r") as f:
        data = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in json.load(f).items()}

    return data


def log_grid(ax: plt.Axes):
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")


def proc_plot_traj(ax: plt.Axes, data, name: str, cumtime_key: str, color: str, linestyle: str, label: str):
    time_step, traj = get_performance_over_time(cumtimes=data[name][cumtime_key], perf_vals=data[name]["loss"])
    line = plot_traj(ax=ax, time_step=time_step, traj=traj, color=color, label=label, linestyle=linestyle)
    if cumtime_key == "actual_cumtime":
        tmins, tmaxs, ymin, ymax = get_tmins_and_tmaxs_and_yrange(data)
        ax.vlines(time_step[-1], ymin, ymax, color=color, linestyle="dashed")

    return line, label, time_step[0], time_step[-1]


def plot_naive(ax: plt.Axes, deterministic: bool):
    suffix = "deterministic" if deterministic else "noisy"
    data = get_data(suffix)
    line, label, t0, t1 = proc_plot_traj(
        ax, data=data, name="naive", cumtime_key="actual_cumtime", color="black", linestyle="dotted", label="Naïve"
    )
    return line, label, t0, t1


def plot_perf_over_time(
    ax: plt.Axes, deterministic: bool, ours_key: str, ours_label: str, ours_color: str, add_naive: bool
):
    suffix = "deterministic" if deterministic else "noisy"
    data = get_data(suffix)

    tmin, tmax = np.inf, -np.inf
    kwargs = dict(ax=ax, data=data, name=ours_key, color=ours_color)
    lines, labels = [], []
    for cumtime_key, label, linestyle in zip(
        ["actual_cumtime", "simulated_cumtime"],
        [ours_label, f"Reproduced from {ours_label}"],
        [None, "dotted"],
    ):
        line, label, t0, t1 = proc_plot_traj(**kwargs, cumtime_key=cumtime_key, linestyle=linestyle, label=label)
        tmin, tmax = min(tmin, t0), max(tmax, t1)
        lines.append(line)
        labels.append(label)

    if add_naive:
        line, label, t0, t1 = plot_naive(ax, deterministic)
        tmin, tmax = min(tmin, t0), max(tmax, t1)
        lines.append(line)
        labels.append(label)

    _, tmaxs, ymin, ymax = get_tmins_and_tmaxs_and_yrange(data)
    diff = ymax - ymin
    factor = 1e-1 if ours_key == "ours" else 5e-2
    target_pos = (tmaxs[ours_key], ymax - factor * diff)
    add_arrow(
        ax,
        src_pos=(tmaxs["naive"], ymax - factor * diff),
        target_pos=target_pos,
        color=ours_color,
        linestyle="dashed",
    )
    speedup = int(tmaxs["naive"] / float(f"{tmaxs[ours_key]:.0e}"))
    ax.text(
        target_pos[0] * 0.7,
        target_pos[1],
        f"{speedup}x",
        horizontalalignment="right",
        verticalalignment="top",
        bbox=dict(facecolor="white", lw=0),
        zorder=10,
    )
    ax.set_xlim(right=max(t for t in tmaxs.values()) * 1.1)
    ax.set_ylim(ymin - 1e-2 * diff, ymax + 1e-2 * diff)
    ax.set_xscale("log")
    log_grid(ax)
    return lines, labels


if __name__ == "__main__":
    fmt = "pdf"
    fig, axes = plt.subplots(
        figsize=(15, 5),
        ncols=2,
        sharex=True,
        sharey=True,
        gridspec_kw=dict(wspace=0.03),
    )
    fig.supxlabel("Wall-Clock Time [s]", y=-0.05)
    fig.supylabel("Cumulative Minimum Function Value", x=0.065, y=0.47)
    axes[0].set_title("Deterministic Objective")
    axes[1].set_title("Noisy Objective")
    lines, labels = [], []
    for deterministic in [True, False]:
        _lines, _labels = plot_perf_over_time(
            ax=axes[1 - deterministic],
            deterministic=deterministic,
            ours_key="ours",
            ours_label="Ours",
            ours_color="red",
            add_naive=False,
        )
        if not deterministic:
            lines += _lines
            labels += _labels
        _lines, _labels = plot_perf_over_time(
            ax=axes[1 - deterministic],
            deterministic=deterministic,
            ours_key="ours_ask_and_tell",
            ours_label="Ours (A&T)",
            ours_color="blue",
            add_naive=True,
        )
        if not deterministic:
            lines += _lines
            labels += _labels

    axes[0].legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(1.05, -0.2),
        ncol=3,
    )
    plt.savefig(f"demo/validation{MODE}.{fmt}", bbox_inches="tight")
