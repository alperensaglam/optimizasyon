"""
experiments/cbc_vs_greedy.py

CBC vs Greedy comparison across instance sizes.

Runs both solvers on subsets of the Istanbul neighborhood data, records
runtime, objective value, number of open DCs, and optimality gap.

Output: results/cbc_vs_greedy.csv
"""

import os
import sys

import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from model.cflp import solve_cflp

# ── parameters ────────────────────────────────────────────────────────────────

ALPHA = 1.675        # optimal alpha from Furkan's golden-section search
Q     = 200_000.0    # DC capacity
Q0    = 100_000.0    # reference capacity
CBC_TIME_LIMIT = 120 # seconds per CBC run

# Instance sizes to test: small → medium → large
# CBC is skipped for n>400 (too many variables for a 120s limit)
INSTANCE_SIZES = [20, 50, 100, 200, 400, 890]
CBC_MAX_N      = 400   # skip CBC for instances larger than this

# ── data loading ──────────────────────────────────────────────────────────────

DATA_DIR    = os.path.join(_BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(_BASE_DIR, "results")

print("Loading data...")
df_n   = pd.read_csv(os.path.join(DATA_DIR, "neighborhoods.csv"))
t_full = np.load(os.path.join(DATA_DIR, "travel_times.npy"))
w_full = df_n["population"].values.astype(float)
r_full = df_n["rent_per_m2"].values.astype(float)

print(f"Loaded {len(w_full)} neighborhoods, travel matrix {t_full.shape}")
print(f"Parameters: alpha={ALPHA}, Q={Q:,.0f}, Q0={Q0:,.0f}, CBC limit={CBC_TIME_LIMIT}s\n")

# ── run experiments ───────────────────────────────────────────────────────────

rows = []

for n in INSTANCE_SIZES:
    w = w_full[:n]
    t = t_full[:n, :n]
    r = r_full[:n]

    for method in ("greedy", "cbc"):
        if method == "cbc" and n > CBC_MAX_N:
            print(f"  n={n:4d}  {method:6s}  SKIPPED (n > {CBC_MAX_N})")
            continue

        print(f"  n={n:4d}  {method:6s}  ", end="", flush=True)
        res = solve_cflp(w, t, r, ALPHA, Q, Q0, method=method,
                         time_limit=CBC_TIME_LIMIT)

        # Ensure gap is a proper float (NaN if not available, not empty)
        gap_val = res["gap"]
        if gap_val is None or (isinstance(gap_val, float) and np.isnan(gap_val)):
            gap_val = float("nan")

        row = {
            "instance_size":   n,
            "method":          method,
            "runtime_s":       round(res["runtime"], 3),
            "objective_value": round(res["objective"], 2),
            "n_open_dcs":      int(res["y"].sum()),
            "gap":             gap_val,
        }
        rows.append(row)
        print(f"runtime={row['runtime_s']:.2f}s  obj={row['objective_value']:.0f}  "
              f"n_open={row['n_open_dcs']}  gap={row['gap']:.4f}" if not np.isnan(row["gap"])
              else f"runtime={row['runtime_s']:.2f}s  obj={row['objective_value']:.0f}  "
                   f"n_open={row['n_open_dcs']}  gap=nan")

# ── save results ──────────────────────────────────────────────────────────────

df_out = pd.DataFrame(rows)
os.makedirs(RESULTS_DIR, exist_ok=True)
out_path = os.path.join(RESULTS_DIR, "cbc_vs_greedy.csv")
df_out.to_csv(out_path, index=False)
print(f"\nSaved → {out_path}")

# ── summary table ──────────────────────────────────────────────────────────────

print("\nResults:")
print(df_out.to_string(index=False))

# Objective gap between methods (where both ran)
print("\nObjective ratio (greedy / CBC) for matched instances:")
grp = df_out.pivot(index="instance_size", columns="method", values="objective_value")
if "cbc" in grp and "greedy" in grp:
    grp["greedy_ratio"] = grp["greedy"] / grp["cbc"]
    print(grp[["cbc", "greedy", "greedy_ratio"]].dropna().to_string())
