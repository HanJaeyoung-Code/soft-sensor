"""
step1_soft_sensor.py — a PLS soft sensor for penicillin titer.

Predicts the slow offline lab value (penicillin g/L) from the online sensors,
and is validated on whole HELD-OUT batches so the score reflects performance
on a brand-new fermentation run, not memorised rows.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import r2_score, mean_squared_error

from step0_data_loader import load_timeseries, sensor_columns

# --- 1. Pull the tidy data from the spine ------------------------------
df = load_timeseries()
sensors = sensor_columns(df)
df = df.dropna(subset=sensors + ["penicillin"]).copy()
df = df.sort_values(["batch", "time"]).reset_index(drop=True)

# --- 1b. Memory features: the "odometer" the instantaneous sensors lack -
# Same instantaneous readings can mean "early-quiet" or "late-quiet" --
# cumulative inputs break that ambiguity because total input tracks total
# output. "time" is already elapsed hours within the batch; add cumulative
# sugar fed and cumulative OUR (running sums, not dt-weighted integrals).
sugar_col = next(c for c in sensors if "Sugar feed rate" in c)
our_col   = next(c for c in sensors if "Oxygen Uptake Rate" in c)
df["cum_sugar_fed"] = df.groupby("batch")[sugar_col].cumsum()
df["cum_OUR"]       = df.groupby("batch")[our_col].cumsum()

feature_cols = sensors + ["time", "cum_sugar_fed", "cum_OUR"]

X = df[feature_cols].values
y = df["penicillin"].values
groups = df["batch"].values          # the label that keeps each batch intact

# --- 2. Hold out WHOLE batches for the final test (the honesty move) ----
splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
g_train = groups[train_idx]

# --- 3. Scale — learn the scaling from TRAIN only ----------------------
xs, ys = StandardScaler(), StandardScaler()
X_train_s = xs.fit_transform(X_train)
X_test_s  = xs.transform(X_test)                     # apply, don't re-learn
y_train_s = ys.fit_transform(y_train.reshape(-1, 1)).ravel()

# --- 4. Choose #components honestly, via batch-grouped cross-validation -
best_k, best_score = 1, -np.inf
gkf = GroupKFold(n_splits=5)
for k in range(1, min(len(feature_cols), 15) + 1):
    scores = []
    for tr, va in gkf.split(X_train_s, y_train_s, g_train):
        m = PLSRegression(n_components=k).fit(X_train_s[tr], y_train_s[tr])
        scores.append(r2_score(y_train_s[va], m.predict(X_train_s[va]).ravel()))
    if np.mean(scores) > best_score:
        best_k, best_score = k, np.mean(scores)
print(f"Chosen components: {best_k}  (CV R2 = {best_score:.3f})")

# --- 5. Fit the final model on all the training data -------------------
pls = PLSRegression(n_components=best_k).fit(X_train_s, y_train_s)

# --- 6. Predict the held-out batches, converted back to real g/L -------
y_pred = ys.inverse_transform(pls.predict(X_test_s).reshape(-1, 1)).ravel()
y_pred = np.clip(y_pred, 0, None)    # penicillin concentration can't be negative

# --- 7. Score on batches the model has never seen ----------------------
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"HELD-OUT batches | R2 = {r2:.3f} | RMSE = {rmse:.3f} g/L")

# --- 8. Which sensors drive the prediction? ----------------------------
# On standardised inputs, each PLS coefficient = that sensor's pull on titer.
importance = sorted(zip(feature_cols, pls.coef_.ravel()),
                    key=lambda t: abs(t[1]), reverse=True)
print("\nTop drivers (standardised coefficient):")
for name, coef in importance[:6]:
    print(f"  {coef:+.3f}  {name}")

# --- 9. Two plots: parity + one real batch traced through time ---------
fig, (axp, axt) = plt.subplots(1, 2, figsize=(12, 5))

axp.scatter(y_test, y_pred, s=6, alpha=0.3)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
axp.plot(lims, lims, "r--", lw=1)
axp.set_xlabel("Measured penicillin (g/L)")
axp.set_ylabel("Soft-sensor prediction (g/L)")
axp.set_title(f"Parity  (R\u00b2 = {r2:.2f})")

test_df = df.iloc[test_idx].copy()
test_df["y_true"], test_df["y_hat"] = y_test, y_pred
b   = test_df["batch"].iloc[0]
sub = test_df[test_df["batch"] == b].sort_values("time")
axt.plot(sub["time"], sub["y_true"], lw=2, label="measured")
axt.plot(sub["time"], sub["y_hat"], lw=1.5, ls="--", label="soft-sensor")
axt.set_xlabel("Time (h)"); axt.set_ylabel("Penicillin (g/L)")
axt.set_title(f"Batch {int(b)} — live tracking"); axt.legend()

plt.tight_layout()
plt.savefig("soft_sensor_results.png", dpi=150)
plt.show()