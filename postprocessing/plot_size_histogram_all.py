from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)

    control_map = mpl.colormaps["winter"]
    rheb_map = mpl.colormaps["spring"]

    cc = 300
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

            cind = i / 5 if exp.startswith('control') else (i - 5) / 5
            color = control_map(cind) if exp.startswith("control") else rheb_map(cind)

            ax.hist(
                stats["cnts"],
                bins=100,
                range=(0, 1000),
                histtype="step",
                alpha=0.8,
                label=f"{exp}",
                color=color,
            )
        ax.legend()
        ax.set_xlabel("Size of objects identified (in number of voxels)")
        ax.set_ylabel("Number of objects falling in the size range")
        ax.set_title(f"threshold={thr}")
        plt.tight_layout()
        fig.savefig(f"size_histogram_{thr}.png")


if __name__ == "__main__":
    tyro.cli(main)
