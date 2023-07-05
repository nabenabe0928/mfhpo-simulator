from __future__ import annotations

import json

import matplotlib.pyplot as plt

import numpy as np


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font


def get_traj(data: dict, name: str, cumtime_key: str) -> tuple[np.ndarray, np.ndarray]:
    n_seeds = len(data["ours"]["loss"])
    tmin = np.max([np.min(t) for t in data[name][cumtime_key]])  # max-min is correct!
    tmax = np.max([np.max(t) for t in data[name][cumtime_key]])
    time_step = np.exp(np.linspace(np.log(tmin), np.log(tmax), 100))

    loss_vals = []
    for i in range(n_seeds):
        loss, cumtime = np.minimum.accumulate(data[name]["loss"][i]), data[name][cumtime_key][i]
        indices = np.searchsorted(cumtime, time_step, side="left")
        loss_vals.append(loss[np.minimum(loss.size - 1, indices)])

    return np.array(loss_vals), time_step


def plot_traj(
    ax: plt.Axes,
    time_step: np.ndarray,
    traj: np.ndarray,
    color: str,
    label: str,
    linestyle: str | None = None,
    marker: str | None = None,
):
    n_seeds = traj.shape[0]
    m, s = np.mean(traj, axis=0), np.std(traj, axis=0) / np.sqrt(n_seeds)
    line = ax.plot(
        time_step,
        m,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markevery=10,
        markersize=10,
        label=label,
    )[0]
    ax.fill_between(time_step, m - s, m + s, color=color, alpha=0.2)
    return line


def main(deterministic: bool, ours_key: str, ours_label: str, file_suffix: str, fmt: str):
    suffix = "deterministic" if deterministic else "noisy"
    with open(f"demo/validation-results-{suffix}.json", mode="r") as f:
        data = {k: {k2: np.array(v2) for k2, v2 in v.items()} for k, v in json.load(f).items()}

    _, ax = plt.subplots(figsize=(10, 5))
    lines, labels = [], []
    tmin, tmax = np.inf, -np.inf
    for name, cumtime_key, color, label, linestyle in zip(
        [ours_key, ours_key, "naive"],
        ["actual_cumtime", "simulated_cumtime", "actual_cumtime"],
        ["red", "red", "blue"],
        [ours_label, f"Reproduced from {ours_label}", "Naïve"],
        [None, "dashed", "dotted"],
    ):
        traj, time_step = get_traj(data, name=name, cumtime_key=cumtime_key)
        tmin, tmax = min(tmin, time_step[0]), max(tmax, time_step[-1])
        line = plot_traj(ax=ax, time_step=time_step, traj=traj, color=color, label=label, linestyle=linestyle)
        lines.append(line)
        labels.append(label)

    ax.set_xlim(tmin, tmax)
    ax.set_xscale("log")
    ax.set_xlabel("Wall-Clock Time [s]")
    ax.set_ylabel("Cumulative Minimum Function Value")
    ax.legend(
        handles=lines,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.45, -0.17),
        ncol=len(labels),
    )
    ax.grid(which="minor", color="gray", linestyle=":")
    ax.grid(which="major", color="black")
    plt.savefig(f"demo/validation-{file_suffix}-{suffix}.{fmt}", bbox_inches="tight")


if __name__ == "__main__":
    kwargs = {"fmt": "png"}
    for c in [True, False]:
        kwargs.update(deterministic=c)
        main(**kwargs, ours_key="ours", ours_label="Ours", file_suffix="ours")
        main(**kwargs, ours_key="ours_ask_and_tell", ours_label="Ours (A&T)", file_suffix="ours-ask-and-tell")
