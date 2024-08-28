from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)
    output_dir = Path(input_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    control_map = mpl.colormaps["winter"]
    rheb_map = mpl.colormaps["spring"]

    for cc in [200]:
        for thr in [127]:
            fig, axs = plt.subplots(4, 4, figsize=(40,40))
            for k in range(1, 15):
                x = k // 4
                y = k % 4
                ax = axs[x, y]
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
                    file = f"dist_{cc}_{thr}.npz"
                    stats = np.load(root_dir / exp / file)

                    cind = i / 5 if exp.startswith("control") else (i - 5) / 5
                    color = (
                        control_map(cind) if exp.startswith("control") else rheb_map(cind)
                    )

                    data = stats["distances"][:, k]
                    ax.hist(
                        data,
                        bins=100,
                        range=(0, 150),
                        density=False,
                        histtype="step",
                        alpha=0.8,
                        label=exp,
                        color=color,
                    )
                ax.legend()
                ax.set_ylabel('')  # Remove the y-label
                ax.set_yticklabels([])
                # ax.set_xlabel(
                #     "Distance between the centroid of an object and that of its nearest neighbor"
                # )
                # ax.set_ylabel("Number of objects falling in the distance range")
                ax.set_title(f"rank {k}")
            plt.tight_layout()
            fig.savefig(output_dir / f"distance_histogram_big_{cc}_{thr}.png")


if __name__ == "__main__":
    tyro.cli(main)
