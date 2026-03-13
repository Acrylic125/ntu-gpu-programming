import argparse
import math
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def infer_grid_size(path: Path) -> int:
    stem = path.stem
    marker = "sim_"
    if not stem.startswith(marker):
        raise ValueError(f"Cannot infer Lk from filename: {path.name}")

    suffix = stem[len(marker):]
    lk_text = suffix.split("_", 1)[0]
    return int(lk_text)


def load_sim(path: Path):
    """Load sim file: episodes separated by blank lines, each episode one flat grid."""
    lk = infer_grid_size(path)
    n = lk * 100

    with path.open() as f:
      content = f.read()

    episodes = []
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        values = np.array([float(x) for x in block.split()])
        if values.size != n * n:
            continue
        episodes.append(values.reshape(n, n))

    return lk, episodes


def discover_sim_files(root: Path):
    return sorted(root.rglob("sim/**/*.txt"))


def make_title(path: Path, lk: int) -> str:
    parent = path.parent.name
    label = path.stem.replace("sim_", "Lk=")
    return f"{parent}: {label} (N={lk * 100})"


def main():
    parser = argparse.ArgumentParser(
        description="Animate all simulation outputs and export them as one MP4."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Root directory to search recursively for sim/**/*.txt files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "combined_simulation.mp4",
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for the exported MP4.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=100,
        help="Preview interval in milliseconds per frame.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the animation window after saving the MP4.",
    )
    args = parser.parse_args()

    sim_files = discover_sim_files(args.root)
    if not sim_files:
        raise SystemExit(f"No simulation files found under {args.root}")

    simulations = []
    for path in sim_files:
        lk, frames = load_sim(path)
        if not frames:
            continue
        simulations.append(
            {
                "path": path,
                "lk": lk,
                "frames": frames,
                "title": make_title(path, lk),
                "vmin": min(frame.min() for frame in frames),
                "vmax": max(frame.max() for frame in frames),
            }
        )

    if not simulations:
        raise SystemExit(f"No valid simulation frames found under {args.root}")

    frame_count = max(len(sim["frames"]) for sim in simulations)
    cols = math.ceil(math.sqrt(len(simulations)))
    rows = math.ceil(len(simulations) / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    axes = np.atleast_1d(axes).ravel()

    artists = []
    for ax, sim in zip(axes, simulations):
        first_frame = sim["frames"][0]
        im = ax.imshow(
            first_frame,
            cmap="hot",
            aspect="equal",
            vmin=sim["vmin"],
            vmax=sim["vmax"],
        )
        ax.set_title(sim["title"])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        artists.append(im)

    for ax in axes[len(simulations):]:
        ax.axis("off")

    def update(frame_idx):
        updated = []
        for im, sim in zip(artists, simulations):
            frames = sim["frames"]
            current = frames[min(frame_idx, len(frames) - 1)]
            im.set_data(current)
            updated.append(im)
        return updated

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_count,
        interval=args.interval,
        blit=True,
        repeat=False,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=args.fps)
    ani.save(args.output, writer=writer)
    print(f"Saved MP4 to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
