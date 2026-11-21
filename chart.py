"""Re-run the 5-fold evaluation per history length and draw docs/accuracy.png.

    uv run python chart.py
"""

import matplotlib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_predict

from train import new_model, training_set

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

WINDOWS = [3, 5, 10, 20, 30, None]
ACCENT = "#c2410c"


def main():
    rows = []
    for days in WINDOWS:
        X, y = training_set(days)
        pred = cross_val_predict(new_model(), X, y, cv=5)
        rows.append({"days": days or "full", "objects": len(X),
                     "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="white")
    x = range(len(df))
    ax.plot(x, df.macro_f1, "o-", color=ACCENT, lw=2.5, ms=8, label="macro F1")
    ax.plot(x, df.accuracy, "o--", color="#9ca3af", lw=1.5, ms=5, label="accuracy")
    for i, v in zip(x, df.macro_f1):
        ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points", xytext=(0, -18), ha="center",
                    color=ACCENT, fontsize=10)
    ax.set_xticks(list(x), [f"first {d} days" if d != "full" else "full history" for d in df.days], fontsize=9)
    ax.set_ylim(min(df.macro_f1.min(), df.accuracy.min()) - 0.06, 1.0)
    ax.set_ylabel("score (5-fold)")
    ax.set_title("How well it sorts objects vs. how much light curve it sees", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig("docs/accuracy.png", dpi=110)


if __name__ == "__main__":
    main()
