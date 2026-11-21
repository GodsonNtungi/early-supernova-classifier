"""Score last night's unclassified ZTF flags and print the top supernova candidates.

    uv run python tonight.py [days_back=10] [top=20]

trains fresh each run (~20 s) instead of pickling a model.
"""

import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from fetch import API, fetch_detections, session
from train import features, new_model, training_set

DAYS_BACK = float(sys.argv[1]) if len(sys.argv) > 1 else 10
TOP = int(sys.argv[2]) if len(sys.argv) > 2 else 20
MJD_EPOCH = dt.datetime(1858, 11, 17, tzinfo=dt.UTC)


def mjd_now() -> float:
    return (dt.datetime.now(dt.UTC) - MJD_EPOCH).total_seconds() / 86400


def query_objects(**extra) -> pd.DataFrame:
    now = mjd_now()
    rows, page = [], 1
    while True:
        r = session.get(f"{API}/objects/", params={
            "firstmjd": [now - DAYS_BACK, now], "ndet": [3, 100],
            "page": page, "page_size": 500, "order_by": "firstmjd", "order_mode": "DESC", **extra},
            timeout=120)
        r.raise_for_status()
        j = r.json()
        rows += j["items"]
        if not j["has_next"]:
            break
        page += 1
    return pd.DataFrame(rows).set_index("oid")


def fresh_objects() -> pd.DataFrame:
    fresh = query_objects()[["ndet", "firstmjd", "lastmjd", "stellar", "meanra", "meandec"]]
    # ALeRCE's image classifier flags camera artefacts; we only use light curves so borrow that check.
    bogus = query_objects(classifier="stamp_classifier", **{"class": "bogus"}).index
    fresh["bogus_image"] = fresh.index.isin(bogus)
    return fresh


def main():
    # Train on light curves cut to the same age window the fresh flags have.
    X, y = training_set(days=DAYS_BACK)
    clf = new_model().fit(X, y)

    fresh = fresh_objects()
    print(f"{len(fresh)} objects first seen in the last {DAYS_BACK:g} days, fetching...", flush=True)
    with ThreadPoolExecutor(4) as pool:
        det = pd.DataFrame([d for c in pool.map(fetch_detections, fresh.index) for d in c])

    F = features(det).reindex(columns=X.columns).fillna(-99)
    proba = pd.DataFrame(clf.predict_proba(F), index=F.index, columns=clf.classes_)
    out = fresh.join(proba, how="inner")
    out["age_days"] = (out.lastmjd - out.firstmjd).round(1)
    out["nights"] = det.groupby("oid").mjd.apply(lambda s: s.round().nunique()).reindex(out.index)
    out["link"] = "https://alerce.online/object/" + out.index
    out = out.sort_values("supernova", ascending=False)
    # stellar = sits on a known star; one-night-only = could be an asteroid passing through
    keep = out[~out.stellar & ~out.bogus_image & (out.nights >= 2)]
    print(f"\n{len(out)} scored, {out.stellar.sum()} on known stars, {out.bogus_image.sum()} bogus images, "
          f"{(out.nights < 2).sum()} single-night, {len(keep)} kept")

    cols = ["supernova", "variable_star", "agn", "ndet", "nights", "age_days", "link"]
    pd.set_option("display.width", 200)
    print(f"\ntop {TOP} supernova candidates\n")
    print(keep[cols].head(TOP).round(2).to_string())
    out.to_csv("data/tonight.csv")
    det[det.oid.isin(keep.index)].to_parquet("data/tonight_detections.parquet", index=False)
    print("\nfull table -> data/tonight.csv   (run viz.py for a picture)")


if __name__ == "__main__":
    main()
