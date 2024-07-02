from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)

    for exp in ["control2", "control3", "rheb1", "rheb2"]:
        fig, ax = plt.subplots()
        cc = 300
        for thr in [30, 60, 127]:
            file = f"stat_{cc}_{thr}.npz"
            stats = np.load(root_dir / exp / file)
            ax.hist(
                stats["cnts"],
                bins=100,
                range=(200, 1000),
                histtype="step",
                alpha=0.8,
                label=f"{thr}",
            )
        ax.legend()
        ax.set_xlabel('Size of objects identified (in number of voxels)')
        ax.set_ylabel('Number of objects falling in the size range')
        ax.set_title(exp)
        plt.tight_layout()
        fig.savefig(root_dir / exp / f"size_histogram_{exp}.png")


if __name__ == "__main__":
    tyro.cli(main)
