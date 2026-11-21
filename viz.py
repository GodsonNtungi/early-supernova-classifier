"""Plot tonight's top supernova candidates: light curve + model confidence.

    uv run python viz.py [top=12]   ->  data/tonight.png
"""

import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 12
COLOURS = {"supernova": "#d62728", "variable_star": "#1f77b4", "agn": "#7f7f7f"}


def main():
    out = pd.read_csv("data/tonight.csv", index_col="oid")
    det = pd.read_parquet("data/tonight_detections.parquet")
    keep = out[~out.stellar & ~out.bogus_image & (out.nights >= 2)].head(TOP)

    ncol = 3
    nrow = -(-len(keep) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow), squeeze=False)
    for ax, (oid, row) in zip(axes.flat, keep.iterrows()):
        g = det[det.oid == oid]
        t0 = g.mjd.min()
        for fid, c in [(1, "green"), (2, "red")]:
            s = g[g.fid == fid]
            ax.errorbar(s.mjd - t0, s.mag, s.err, fmt="o", color=c, ms=5, alpha=0.8)
        ax.invert_yaxis()
        ax.set_xlabel("days since first alert")
        ax.set_ylabel("mag")
        ax.set_title(oid, fontsize=10, loc="left")
        # confidence bar across the top, in axes coordinates
        x = 0.0
        for cls, c in COLOURS.items():
            w = row[cls]
            ax.axhspan(0, 0, 0, 0)  # no-op keeps autoscale stable
            ax.add_patch(plt.Rectangle((x, 1.02), w, 0.06, transform=ax.transAxes, color=c, clip_on=False))
            x += w
        ax.text(1.0, 1.1, f"SN {row.supernova:.0%}", transform=ax.transAxes, ha="right", fontsize=9, color="#d62728")
    for ax in axes.flat[len(keep):]:
        ax.axis("off")
    fig.suptitle(f"Tonight's top {len(keep)} supernova candidates  "
                 f"(bar: red = supernova, blue = variable star, grey = active galaxy)", y=1.0)
    fig.tight_layout()
    fig.savefig("data/tonight.png", dpi=110, bbox_inches="tight")
    print("data/tonight.png")


if __name__ == "__main__":
    main()
