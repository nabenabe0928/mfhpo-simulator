import os

from benchmark_simulator.utils import get_mean_and_standard_error, get_performance_over_time_from_paths

import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 18
plt.rcParams["mathtext.fontset"] = "stix"  # The setting of math font
opt_labels = {"bohb": "BOHB (HpBandSter)", "dehb": "DEHB", "neps": "NePS", "smac": "SMAC3"}

_, ax = plt.subplots(figsize=(10, 5))

for opt in ["bohb", "dehb", "neps", "smac"]:
    path = f"mfhpo-simulator-info/{opt}/bench=hpolib_dataset=slice-localization_nworkers=4/"
    paths = [os.path.join(path, str(seed)) for seed in range(10)]
    cumtimes, loss_vals = get_performance_over_time_from_paths(paths=paths, obj_key="loss")
    m, s = get_mean_and_standard_error(loss_vals)
    ax.plot(cumtimes, m, label=opt_labels[opt])
    ax.fill_between(cumtimes, m - s, m + s, alpha=0.2)

ax.set_title("Slice Localization (HPOLib) / $\\mathtt{n\\_workers=4}$")
ax.set_xscale("log")
ax.set_xlabel("Wall-Clock Time [s]")
ax.set_ylabel("Cumulative Minimum Function Value")
ax.grid(which="minor", color="gray", linestyle=":")
ax.grid(which="major", color="black")
ax.legend()
plt.savefig("comparison-example.png", bbox_inches="tight")
