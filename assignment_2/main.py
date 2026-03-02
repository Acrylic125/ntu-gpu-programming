import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_sim(path: str, Lk: int):
    """Load sim file: episodes separated by \\n\\n, each episode one line of space-separated floats."""
    with open(path) as f:
        content = f.read()
    N = Lk * 100  # gridPointsOnAxis from p1.cu
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


def main():
    parser = argparse.ArgumentParser(description="Animate wave simulation heatmap")
    parser.add_argument("Lk", type=int, help="Domain size (loads ./sim/sim_{Lk}.txt)")
    args = parser.parse_args()
    Lk = args.Lk
    path = f"./sim/sim_{Lk}.txt"

    frames = load_sim(path, Lk)
    if not frames:
        raise SystemExit(f"No episodes found in {path}")

    fig, ax = plt.subplots()
    vmin = min(f.min() for f in frames)
    vmax = max(f.max() for f in frames)
    im = ax.imshow(frames[0], cmap="hot", aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax)
    ax.set_title(f"Wave simulation Lk={Lk}")

    def update(frame_idx):
        im.set_data(frames[frame_idx])
        return [im]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=50,
        blit=True,
        repeat=True,
    )
    plt.show()


if __name__ == "__main__":
    main()
