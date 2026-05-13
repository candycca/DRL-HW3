from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import imageio

COLOR_MAP = {
    " ": np.array([255, 255, 255], dtype=np.uint8),
    "P": np.array([66, 135, 245], dtype=np.uint8),
    "+": np.array([50, 168, 82], dtype=np.uint8),
    "-": np.array([220, 53, 69], dtype=np.uint8),
    "W": np.array([120, 120, 120], dtype=np.uint8),
}


def plot_loss(metrics, output_path: str, title: Optional[str] = None) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["episode"], metrics["loss"], label="loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_loss_compare(experiments: Dict[str, np.ndarray], output_path: str) -> None:
    plt.figure(figsize=(6, 4))
    for name, metrics in experiments.items():
        plt.plot(metrics["episode"], metrics["loss"], label=name)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def render_board(board_array: np.ndarray, scale: int = 32) -> np.ndarray:
    h, w = board_array.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            token = str(board_array[i, j])
            image[i, j] = COLOR_MAP.get(token, COLOR_MAP[" "])
    image = np.kron(image, np.ones((scale, scale, 1), dtype=np.uint8))
    return image


def make_dashboard_gif(frames: Iterable[np.ndarray], output_path: str, fps: int = 4) -> None:
    imageio.mimsave(output_path, list(frames), fps=fps, loop=0)
