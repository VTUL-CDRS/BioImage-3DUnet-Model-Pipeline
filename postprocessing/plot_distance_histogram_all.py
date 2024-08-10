from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)

    control_map = mpl.colormaps["winter"]
    rheb_map = mpl.colormaps["spring"]

    for cc in [300, 500]:
        for thr in [30, 60, 127]:
            fig, ax = plt.subplots()
            for i, exp in enumerate(
                [
                    "control1",
                    "control2",
                    "control3",
                    "control4",
                    "control8",
                    "rheb1",
                    "rheb2",
                    "rheb3",
                    "rheb4",
                ]
            ):
                file = f"stat_{cc}_{thr}.npz"
                stats = np.load(root_dir / exp / file)

                cind = i / 5 if exp.startswith("control") else (i - 5) / 5
                color = (
                    control_map(cind) if exp.startswith("control") else rheb_map(cind)
                )

                ax.hist(
                    stats["nn_distances"],
                    bins=100,
                    range=(0, 80),
                    histtype="step",
                    alpha=0.8,
                    label=exp,
                    color=color,
                )
            ax.legend()
            ax.set_xlabel(
                "Distance between the centroid of an object and that of its nearest neighbor"
            )
            ax.set_ylabel("Number of objects falling in the distance range")
            ax.set_title(f"{cc}_{thr}")
            plt.tight_layout()
            fig.savefig(f"distance_histogram_{cc}_{thr}.png")


if __name__ == "__main__":
    tyro.cli(main)
