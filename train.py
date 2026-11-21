"""Hand-made light-curve features + random forest. Prints a report.

12 features per band and a forest. Upgrade to a sequence model
only if this plateaus well below the ALeRCE published numbers (~0.9 macro F1).
"""

import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_predict

warnings.filterwarnings("ignore", category=np.exceptions.RankWarning)  # same-night repeats make polyfit grumble


def band_features(g: pd.DataFrame) -> dict:
    g = g.sort_values("mjd")
    t, m = g.mjd.to_numpy(), g.mag.to_numpy()
    if len(m) < 3:
        return {}
    t = t - t[0]
    peak = m.argmin()  # brightest = smallest magnitude
    f = {
        "n": len(m),
        "span": t[-1],
        "amp": m.max() - m.min(),
        "std": m.std(),
        "skew": pd.Series(m).skew(),
        "rise_time": t[peak],
        "fade_time": t[-1] - t[peak],
        "rise_slope": (m[0] - m[peak]) / max(t[peak], 0.5),
        "fade_slope": (m[-1] - m[peak]) / max(t[-1] - t[peak], 0.5),
        "frac_above_median": (m < np.median(m)).mean(),
        "max_gap": np.diff(t).max(),
        "mean_err": g.err.mean(),
        "first_mag": m[0],
        "peak_mag": m[peak],
        "last_mag": m[-1],
        "peak_is_last": float(peak == len(m) - 1),  # still brightening
        "slope_last3": np.polyfit(t[-3:], m[-3:], 1)[0] if np.ptp(t[-3:]) > 0 else 0.0,  # same-night points crash polyfit
        "curvature": np.polyfit(t, m, 2)[0] if len(m) >= 4 and t[-1] > 0 else 0.0,
        "chi2_const": (((m - m.mean()) / g.err.to_numpy()) ** 2).mean(),  # how un-flat
    }
    return f


def colour_features(det: pd.DataFrame) -> pd.DataFrame:
    """g minus r per night, then its level and change. fids 1=g, 2=r."""
    d = det.assign(night=det.mjd.round())
    nightly = d.groupby(["oid", "night", "fid"]).mag.mean().unstack("fid")
    if not {1, 2} <= set(nightly.columns):
        return pd.DataFrame()
    nightly = nightly.dropna()
    col = (nightly[1] - nightly[2]).rename("gr")
    out = col.groupby("oid").agg(gr_mean="mean", gr_first="first", gr_last="last", gr_nights="size")
    out["gr_change"] = out.gr_last - out.gr_first
    return out


def context_features(det: pd.DataFrame) -> pd.DataFrame:
    """Per-alert context known on night one: sits on a catalogued source? ever dimmer than it?"""
    if "isdiffpos" not in det:
        return pd.DataFrame()
    g = det.groupby("oid")
    return pd.DataFrame({
        "frac_negative": g.isdiffpos.apply(lambda s: (s < 0).mean()),  # dimmer than reference: variables only
        "distnr_median": g.distnr.median(),  # arcsec to nearest reference source
        "frac_on_source": g.corrected.mean(),  # within 1.4 arcsec of a reference source
    })


def features(det: pd.DataFrame) -> pd.DataFrame:
    det = det[det.fid.isin([1, 2])]  # g and r only; the i filter is rare and untrained
    rows = {}
    for (oid, fid), g in det.groupby(["oid", "fid"]):
        rows.setdefault(oid, {}).update({f"{k}_{fid}": v for k, v in band_features(g).items()})
    X = pd.DataFrame.from_dict(rows, orient="index")
    X = X.join(colour_features(det)).join(context_features(det))
    X["g_minus_r"] = X.get("gr_mean")  # kept for test_features / old callers
    return X


def truncate(det: pd.DataFrame, days: float | None) -> pd.DataFrame:
    """Keep only the first `days` after each object's first detection (early classification)."""
    if days is None:
        return det
    return det[det.mjd - det.groupby("oid").mjd.transform("min") <= days]


def training_set(days: float | None = None) -> tuple[pd.DataFrame, pd.Series]:
    det = truncate(pd.read_parquet("data/detections.parquet"), days)
    labels = pd.read_csv("data/labels.csv").set_index("oid")
    X = features(det).fillna(-99)
    return X, labels.label.reindex(X.index)


def new_model() -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=0, n_jobs=-1)


def main():
    days = float(sys.argv[1]) if len(sys.argv) > 1 else None
    X, y = training_set(days)
    print(f"{'first %g days only' % days if days else 'full history'}: "
          f"{len(X)} objects, {X.shape[1]} features\n{y.value_counts().to_string()}\n")

    clf = new_model()
    pred = cross_val_predict(clf, X, y, cv=5)
    print(classification_report(y, pred, digits=3))
    names = sorted(y.unique())
    print(pd.DataFrame(confusion_matrix(y, pred, labels=names), index=names, columns=names))

    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, X.columns).sort_values(ascending=False)
    print("\ntop features\n" + imp.head(8).to_string())


if __name__ == "__main__":
    main()
