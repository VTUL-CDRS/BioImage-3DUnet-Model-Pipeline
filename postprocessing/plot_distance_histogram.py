from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)

    for exp in ["control1", "control2", "control3", "control4", "control8", "rheb1", "rheb2", "rheb3", "rheb4"]:
        fig, ax = plt.subplots()
        for cc in [300, 500]:
            for thr in [30, 60, 127]:
                file = f"stat_{cc}_{thr}.npz"
                stats = np.load(root_dir / exp / file)
                ax.hist(
                    stats["nn_distances"],
                    bins=100,
                    range=(0, 80),
                    histtype="step",
                    alpha=0.8,
                    label=f"{cc}_{thr}",
                )
        ax.legend()
        ax.set_xlabel('Distance between the centroid of an object and that of its nearest neighbor')
        ax.set_ylabel('Number of objects falling in the distance range')
        ax.set_title(exp)
        plt.tight_layout()
        fig.savefig(root_dir / exp / f"distance_histogram_{exp}.png")


if __name__ == "__main__":
    tyro.cli(main)
