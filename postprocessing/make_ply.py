from pathlib import Path

import numpy as np
import tyro
from matplotlib import cm
from plyfile import PlyData, PlyElement


def main(input: str):
    for exp in [
        "control1",
        "control2",
        "control3",
        "control4",
        "control8",
        "rheb1",
        "rheb2",
        "rheb3",
        "rheb4",
    ]:
        file = "stat_200_127.npz"
        stats = np.load(Path(input) / exp / file)
        print(Path(input) / exp / file)
        
        areas = stats["areas"]
        centroids = stats["centroids"]

        mmin, mmax = 400, 600
        areas = (areas - mmin) / (mmax - mmin)

        areas = np.clip(areas, 0.0, 1.0)
        colormap = cm.get_cmap("turbo_r")
        rgb = colormap(areas)
        rgb = (255 * rgb).astype(np.uint8)

        # Create a structured array suitable for PlyElement
        vertex_data = np.zeros(
            centroids.shape[0],
            dtype=[
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ],
        )

        # Assign the location and color data to the structured array
        vertex_data["x"] = centroids[:, 0]
        vertex_data["y"] = centroids[:, 1]
        vertex_data["z"] = centroids[:, 2]
        vertex_data["red"] = rgb[:, 0]
        vertex_data["green"] = rgb[:, 1]
        vertex_data["blue"] = rgb[:, 2]

        # Create the PlyElement
        vertex_element = PlyElement.describe(vertex_data, "vertex")

        # Write to a PLY file
        PlyData([vertex_element]).write(f"{exp}.ply")


if __name__ == "__main__":
    tyro.cli(main)
