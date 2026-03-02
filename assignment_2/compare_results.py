import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_sim(path: str, Lk: int):
    """Load sim file: episodes separated by \\n\\n, each episode one line of space-separated floats."""
    with open(path) as f:
        content = f.read()
    N = Lk * 100
    episodes = []
    for block in content.strip().split("\n\n"):
        block = block.strip()
        if not block:
            continue
        values = np.array([float(x) for x in block.split()])
        if values.size != N * N:
            continue
        episodes.append(values.reshape(N, N))
    return episodes


def load_last_snapshot(path: str, Lk: int):
    """Load last snapshot from file: either single line or multi-episode (\\n\\n separated)."""
    episodes = load_sim(path, Lk)
    if episodes:
        return episodes[-1]
    with open(path) as f:
        line = f.read().strip()
    N = Lk * 100
    values = np.array([float(x) for x in line.split()])
    if values.size != N * N:
        raise ValueError(f"Expected {N * N} values, got {values.size}")
    return values.reshape(N, N)


def main():
    parser = argparse.ArgumentParser(
        description="Compare last snapshot in results/with_results vs results/without_results"
    )
    parser.add_argument("Lk", type=int, nargs="?", default=1, help="Domain size (default: 1)")
    args = parser.parse_args()
    Lk = args.Lk

    with_path = f"results/with_save/sim_{Lk}.txt"
    without_path = f"results/without_save/sim_{Lk}.txt"

    with_grid = load_last_snapshot(with_path, Lk)
    without_grid = load_last_snapshot(without_path, Lk)

    vmin = min(with_grid.min(), without_grid.min())
    vmax = max(with_grid.max(), without_grid.max())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(with_grid, cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
    ax1.set_title(f"results/with_results (last snapshot)")
    ax1.set_xlabel("j")
    ax1.set_ylabel("i")

    im2 = ax2.imshow(without_grid, cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
    ax2.set_title(f"results/without_results (last snapshot)")
    ax2.set_xlabel("j")
    ax2.set_ylabel("i")

    plt.colorbar(im1, ax=[ax1, ax2], shrink=0.6, label="u")
    plt.suptitle(f"Comparison Lk={Lk}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
