"""
step2_early_warning.py — predict a batch's FINAL penicillin titer using only
its first N hours of sensor data. Sweeps N to answer the operator's real
question: "how early can we reliably call how this batch will end up?"
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error

from step0_data_loader import load_timeseries, sensor_columns

# --- 1. Pull the tidy data from the spine, add the cumulative features --
df = load_timeseries()
sensors = sensor_columns(df)
df = df.dropna(subset=sensors + ["penicillin"]).copy()
df = df.sort_values(["batch", "time"]).reset_index(drop=True)

sugar_col = next(c for c in sensors if "Sugar feed rate" in c)
our_col   = next(c for c in sensors if "Oxygen Uptake Rate" in c)
df["cum_sugar_fed"] = df.groupby("batch")[sugar_col].cumsum()
df["cum_OUR"]       = df.groupby("batch")[our_col].cumsum()

# --- 2. The TARGET: each batch's final titer (one fixed number per batch) -
final_titer = df.groupby("batch")["penicillin"].last()

# --- 3. Feature builder: squeeze first N hours into ONE row per batch ----
def make_features(cutoff_h):
    rows = []
    for batch_id, g in df.groupby("batch"):
        window = g[g["time"] <= cutoff_h]          # only what we'd have seen by hour N
        if len(window) < 5:                        # too little to summarise
            continue

        # (a) LEVEL clues — the averages you already had
        feat = {c + "__mean": window[c].mean() for c in sensors}
        feat["cum_sugar_fed"] = window["cum_sugar_fed"].iloc[-1]   # total sugar by hour N
        feat["cum_OUR"]       = window["cum_OUR"].iloc[-1]         # total O2 by hour N

        # (b) SHAPE clues — the new part: is it climbing, and where is it NOW?
        # "last value" = the most recent reading at the cutoff, not an average.
        feat["pen_last"]   = window["penicillin"].iloc[-1]
        feat["sugar_last"] = window[sugar_col].iloc[-1]
        feat["our_last"]   = window[our_col].iloc[-1]

        # "slope" = how fast penicillin is rising over the window (rise / run).
        # np.polyfit(x, y, 1)[0] is the best-fit straight-line slope.
        t = window["time"].values
        if t[-1] > t[0]:                                   # guard against divide-by-zero
            feat["pen_slope"] = np.polyfit(t, window["penicillin"].values, 1)[0]
            feat["our_slope"] = np.polyfit(t, window[our_col].values, 1)[0]
        else:
            feat["pen_slope"] = 0.0
            feat["our_slope"] = 0.0

        feat["batch"] = batch_id
        rows.append(feat)

    X = pd.DataFrame(rows).set_index("batch")
    y = final_titer.loc[X.index]
    return X, y

# --- 4. Sweep the cutoff hour; score each with cross-validation ----------
cutoffs = [20, 40, 60, 80, 100, 120, 140, 160]
model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for N in cutoffs:
    X, y = make_features(N)
    y_cv = cross_val_predict(model, X, y, cv=kf)   # each batch predicted by a model that didn't see it
    r2   = r2_score(y, y_cv)
    rmse = np.sqrt(mean_squared_error(y, y_cv))
    results.append((N, r2, rmse))
    print(f"By hour {N:3d} | R2 = {r2:5.3f} | RMSE = {rmse:5.2f} g/L | batches = {len(y)}")

res = pd.DataFrame(results, columns=["cutoff_h", "r2", "rmse"])

# --- 5. Find the earliest "good enough" decision hour --------------------
THRESH = 0.80
passed = res[res["r2"] >= THRESH]
decision_h = int(passed["cutoff_h"].iloc[0]) if len(passed) else int(res.loc[res.r2.idxmax(), "cutoff_h"])
print(f"\nEarliest hour with R2 >= {THRESH}: {decision_h}")

# --- 6. Two plots: the sweep curve + a parity at the decision hour -------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(res["cutoff_h"], res["r2"], "o-", lw=2)
ax1.axhline(THRESH, color="green", ls=":", label=f"'good enough' (R\u00b2={THRESH})")
ax1.axvline(100, color="red", ls="--", alpha=0.6, label="SPC hour-100 divergence")
ax1.set_xlabel("Hours of batch observed")
ax1.set_ylabel("R\u00b2 predicting FINAL titer")
ax1.set_title("How early can we call the batch?")
ax1.set_ylim(0, 1); ax1.legend()

X, y = make_features(decision_h)
y_cv = cross_val_predict(model, X, y, cv=kf)
ax2.scatter(y, y_cv, s=35, alpha=0.7)
lims = [min(y.min(), y_cv.min()), max(y.max(), y_cv.max())]
ax2.plot(lims, lims, "r--", lw=1)
ax2.set_xlabel("Actual final titer (g/L)")
ax2.set_ylabel("Predicted final titer (g/L)")
ax2.set_title(f"Prediction using only first {decision_h} h  (R\u00b2={r2_score(y, y_cv):.2f})")

plt.tight_layout()
plt.savefig("early_warning_results.png", dpi=150)
plt.show()