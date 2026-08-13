"""This is for visual verification of the synthetic multi-target generator."""

from pathlib import Path
import matplotlib.pyplot as plt

from config import (
    VERIFY_OUTPUT_DIR, VERIFY_SAMPLES, VERIFY_SEED, MAX_TARGETS
)
from env_gen import MultiTargetTrackletDataset


def save_scene(sample, index, out):
    raw = sample["raw_frames"].numpy()
    blip = sample["frames"].numpy()[:,0]
    coords = sample["coords"].numpy()
    mask = sample["target_mask"].numpy() > .5
    n = int(sample["target_count"].item())

    fig, ax = plt.subplots(2, 4, figsize=(14, 7))

    for t in range(4):
        ax[0,t].imshow(raw[t], cmap="gray")
        ax[0,t].set_title(f"RAW F{t}")
        ax[0,t].axis("off")

        ax[1,t].imshow(blip[t], cmap="gray", vmin=-1, vmax=1)
        for k in range(MAX_TARGETS):
            if not mask[k]:
                continue
            p = coords[k]
            ax[1,t].plot(p[:,0], p[:,1], linewidth=1.5)
            ax[1,t].scatter(
                p[t,0], p[t,1],
                s=65, facecolors="none",
                edgecolors="red", linewidths=1.5
            )
            ax[1,t].text(p[t,0]+4, p[t,1]-4, f"T{k}", fontsize=8)

        ax[1,t].set_title(f"BLIP F{t} + ground truth")
        ax[1,t].axis("off")

    fig.suptitle(
        f"V5 Multi-Target | example {index:03d} | targets={n}"
    )
    fig.tight_layout()
    fig.savefig(
        out/f"{index:03d}_combined.png",
        dpi=130, bbox_inches="tight"
    )
    plt.close(fig)


def main():
    out = Path(VERIFY_OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    ds = MultiTargetTrackletDataset(
        VERIFY_SAMPLES, seed=VERIFY_SEED
    )

    counts = {}
    print("="*65)
    print("V5 MULTI-TARGET SYNTHETIC DATA VERIFICATION")
    print("="*65)

    for i in range(VERIFY_SAMPLES):
        s = ds[i]
        n = int(s["target_count"].item())
        counts[n] = counts.get(n, 0) + 1

        print(f"[{i+1:02d}/{VERIFY_SAMPLES}] targets={n}")
        for k in range(n):
            p = s["coords"].numpy()[k]
            print(
                f"  T{k}: "
                f"({p[0,0]:6.1f},{p[0,1]:6.1f}) -> "
                f"({p[3,0]:6.1f},{p[3,1]:6.1f})"
            )
        save_scene(s, i, out)

    print("\nTarget-count distribution:")
    for n in sorted(counts):
        print(f"  {n} target(s): {counts[n]}")
    print(f"\n[SUCCESS] -> {out}")


if __name__ == "__main__":
    main()
