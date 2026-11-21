"""One check: a fake supernova-shaped light curve must yield short rise, long fade."""

import numpy as np
import pandas as pd

from train import features


def test_supernova_shape():
    t = np.arange(0, 60, 2.0)
    mag = 19 - 3 * np.exp(-((t - 10) / 5) ** 2) + 0.02 * (t - 10)  # brightens, then fades slowly
    det = pd.DataFrame({"oid": "X", "mjd": 58000 + t, "fid": 1, "mag": mag, "err": 0.05})
    det = pd.concat([det, det.assign(fid=2, mag=mag + 0.3)])
    X = features(det)
    row = X.loc["X"]
    assert row["rise_time_1"] < row["fade_time_1"]
    assert 3.5 < row["amp_1"] < 4.5  # 3 mag peak plus ~1 mag of linear fade tail
    assert abs(row["g_minus_r"] + 0.3) < 1e-6
    assert row["n_2"] == 30


def test_single_night_does_not_crash():
    det = pd.DataFrame({"oid": "Y", "mjd": 58000.0, "fid": 1, "mag": [19.0, 19.1, 19.2, 19.3], "err": 0.05})
    row = features(det).loc["Y"]
    assert row["slope_last3_1"] == 0 and row["curvature_1"] == 0


if __name__ == "__main__":
    test_supernova_shape()
    test_single_night_does_not_crash()
    print("ok")
