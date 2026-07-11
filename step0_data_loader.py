"""
step0_data_loader.py — the spine for the soft-sensor project.

Single source of truth: which file to read, which columns are the online
sensors (inputs, X), which column is the penicillin target (y), and how a
"batch" is defined on the time axis. Every later step imports load_timeseries()
from here, so no two stages can ever disagree about what we're predicting.
"""
import os
import pandas as pd

# --- 1. What to read (this is the ONLY place you edit column choices) ---
CSV_PATH   = "100_Batches_IndPenSim_V3_1.csv"
CACHE_PATH = "cache_timeseries.parquet"

# The target: the slow offline lab value the soft sensor will predict.
TARGET_KEY = "Penicillin concentration"

# The inputs: sensors that stream continuously in a real plant.
# We match on substrings because IndPenSim's real headers are messy.
SENSOR_KEYS = [
    "Aeration rate", "Agitator RPM", "Sugar feed rate",
    "Acid flow rate", "Base flow rate", "Dissolved oxygen",
    "Temperature", "Vessel Volume", "carbon dioxide percent in off-gas",
    "Oxygen Uptake Rate", "Air head pressure",
]

TIME_KEY  = "Time (h)"
# NOTE: the column literally named "Batch ID" in the raw CSV is a red
# herring -- verified directly against the file, it has ~47k unique values
# (looks like a running sample counter), not a per-batch index. The real
# 1..100 batch axis is the oddly-named "1-Raman spec recorded" column
# (confirmed: exactly 100 unique values). Same trap the SPC project hit
# with "Batch reference", which also looked right but only had 2 values.
BATCH_KEY = "1-Raman spec recorded"

# --- 2. Turn a "concept" into the real (messy) column name -------------
def _resolve(concept, columns, exclude=None):
    """Find the column whose name contains `concept` (case-insensitive)."""
    hits = [c for c in columns if concept.lower() in c.lower()]
    if exclude:                                   # e.g. skip 'Offline Penicillin...'
        hits = [c for c in hits if exclude.lower() not in c.lower()]
    if not hits:
        raise KeyError(f"No column matches '{concept}'. First few: {list(columns)[:8]}")
    return hits[0]


# --- 3. The one public function every later step calls -----------------
def load_timeseries(use_cache=True):
    # 3a. Fast path: if we already built the tidy table, just reload it.
    if use_cache and os.path.exists(CACHE_PATH):
        return pd.read_parquet(CACHE_PATH)

    # 3b. Read ONLY the header (0 rows) to learn the real column names.
    header = pd.read_csv(CSV_PATH, nrows=0).columns

    # 3c. Resolve each concept we need to an actual column name.
    target_col  = _resolve(TARGET_KEY, header, exclude="offline")   # the live P:g/L, not P_offline
    time_col    = _resolve(TIME_KEY, header)
    sensor_cols = [_resolve(k, header) for k in SENSOR_KEYS]
    wanted = sensor_cols + [target_col, time_col]

    has_batch = any(BATCH_KEY.lower() in c.lower() for c in header)
    if has_batch:
        batch_col = _resolve(BATCH_KEY, header)
        wanted.append(batch_col)

    # 3d. Memory-safe read: pull ONLY those columns from the 2.5 GB file.
    df = pd.read_csv(CSV_PATH, usecols=wanted)

    # 3e. Define the batch axis (the single source of truth).
    if has_batch:
        df["batch"] = df[batch_col].astype(int)
    else:
        # No batch column? A new batch starts wherever the clock resets to ~0.
        df["batch"] = (df[time_col].diff() < 0).cumsum() + 1

    # Fail loudly rather than silently mis-defining the batch axis: a real
    # IndPenSim run is ~100 batches, so a resolved column with thousands of
    # unique values (e.g. a sample counter mistaken for a batch ID) must
    # raise, not quietly produce garbage groupings downstream.
    n_batches = df["batch"].nunique()
    if not (2 <= n_batches <= 500):
        raise ValueError(
            f"'{batch_col if has_batch else '(time-reset fallback)'}' resolved to "
            f"{n_batches} unique batch values -- doesn't look like a per-batch "
            "index. Check BATCH_KEY / raw CSV headers."
        )

    # 3f. Rename target + time to clean names, keep only what we use.
    df = df.rename(columns={target_col: "penicillin", time_col: "time"})
    tidy = df[["batch", "time"] + sensor_cols + ["penicillin"]].copy()

    # 3g. Cache the tidy table so every future run is instant.
    tidy.to_parquet(CACHE_PATH, index=False)
    return tidy


# Expose the sensor names so later steps never re-guess them.
def sensor_columns(df):
    return [c for c in df.columns if c not in ("batch", "time", "penicillin")]


if __name__ == "__main__":
    d = load_timeseries()
    print(f"Loaded {d['batch'].nunique()} batches, {len(d):,} rows.")
    print("Sensors:", sensor_columns(d))
    print(d.head())