from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tyro


def main(input_dir: str):
    root_dir = Path(input_dir)

    for exp in ["control2", "control3", "rheb1", "rheb2"]:
        nums = []
        labels = []
        for cc in [300, 500]:
            for thr in [30, 60, 127]:
                file = f'stat_{cc}_{thr}.npz'
                stats = np.load(Path(input_dir) / 'rheb2' / file)
                nums.append(stats['nn_distances'].shape[0])
                labels.append(f'{cc}_{thr}')

        plt.figure()
        plt.bar(labels, nums, label=labels)
        plt.xlabel('Experiments')
        plt.ylabel('Number of identified objects')
        plt.title(exp)
        plt.tight_layout()
        plt.savefig(root_dir / exp / f"number_of_pores_{exp}.png")


if __name__ == "__main__":
    tyro.cli(main)
