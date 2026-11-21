"""Pull labelled ZTF objects and their light curves from the ALeRCE broker.

Writes data/labels.csv (oid, fine_class, label) and data/detections.parquet
(oid, mjd, fid, mag, err).

labels are ALeRCE's own light-curve classifier at high confidence,
not spectroscopic truth. Swap `fetch_labels` for the ZTF BTS CSV
(sites.astro.caltech.edu/ztf/bts) when that host is reachable.
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://api.alerce.online/ztf/v1"
GROUPS = {
    "supernova": ["SNIa", "SNII", "SNIbc", "SLSN"],
    "variable_star": ["RRL", "CEP", "E", "LPV", "DSCT", "Periodic-Other"],
    "agn": ["AGN", "QSO", "Blazar"],
}
MIN_PROB = {"supernova": 0.5, "variable_star": 0.9, "agn": 0.9}  # SN probs rarely reach 0.9
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 200

session = requests.Session()


def fetch_labels() -> pd.DataFrame:
    rows = []
    for label, classes in GROUPS.items():
        for cls in classes:
            r = session.get(
                f"{API}/objects/",
                params={"classifier": "lc_classifier", "class": cls,
                        "probability": MIN_PROB[label], "page_size": PER_CLASS,
                        "ndet": 10, "order_by": "probability", "order_mode": "DESC"},
                timeout=120,
            )
            r.raise_for_status()
            items = r.json()["items"]
            rows += [{"oid": o["oid"], "fine_class": cls, "label": label} for o in items]
            print(f"{cls:15s} {len(items):5d}", flush=True)
    return pd.DataFrame(rows).drop_duplicates("oid")


def fetch_detections(oid: str) -> list[dict]:
    for attempt in range(8):
        try:
            r = session.get(f"{API}/objects/{oid}/detections", timeout=60)
            if r.status_code == 200:
                return [{"oid": oid, "mjd": d["mjd"], "fid": d["fid"],
                         "mag": d["magpsf"], "err": d["sigmapsf"],
                         "isdiffpos": d["isdiffpos"], "distnr": d["distnr"],
                         "corrected": d["corrected"]} for d in r.json()]
            if r.status_code == 404:
                return []
            if r.status_code == 429:  # rate limited: back off
                time.sleep(2 ** attempt)
        except requests.RequestException:
            time.sleep(2 ** attempt)
    print(f"failed {oid}", file=sys.stderr)
    return []


def main():
    Path("data").mkdir(exist_ok=True)
    labels_path, det_path = Path("data/labels.csv"), Path("data/detections.parquet")
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
    else:
        labels = fetch_labels()
        labels.to_csv(labels_path, index=False)
    old = pd.read_parquet(det_path) if det_path.exists() else pd.DataFrame()
    todo = labels.oid[~labels.oid.isin(old.get("oid", []))]
    print(f"{len(labels)} objects, {len(todo)} light curves still to fetch...", flush=True)
    with ThreadPoolExecutor(4) as pool:
        dets = [d for chunk in pool.map(fetch_detections, todo) for d in chunk]
    det = pd.concat([old, pd.DataFrame(dets)], ignore_index=True)
    det.to_parquet(det_path, index=False)
    print(f"{len(det)} detections for {det.oid.nunique()} objects")


if __name__ == "__main__":
    main()
