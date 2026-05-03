"""
optimization/search.py

Outer-layer optimization for CFLP parameters (alpha and Q).
Responsible for finding the "knee point" of the cost-service tradeoff.
"""

import os
import sys
import json
import pandas as pd
import numpy as np

# Add project root to sys.path to allow module imports when running directly
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from model.cflp import solve_cflp

# ── DATA LOADING ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def load_optimization_data():
    """Loads and prepares all data needed for the outer-layer optimization."""
    df_n = pd.read_csv(os.path.join(DATA_DIR, "neighborhoods.csv"))
    t = np.load(os.path.join(DATA_DIR, "travel_times.npy"))
    df_r = pd.read_csv(os.path.join(DATA_DIR, "rents.csv"))
    rent_map = dict(zip(df_r["district"], df_r["avg_rent_per_m2"]))
    
    r = df_n["district"].map(rent_map).values
    w = df_n["population"].values
    return w, t, r

def save_results(data, filename):
    """Saves optimization results to a JSON file for notebook integration."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    print(f"Results saved to: {path}")

# ── LOSS FUNCTION ────────────────────────────────────────────────────────────

def pre_sample_ranges(w, t, r, alpha_range=(0.1, 10.0), Q_range=(50000, 500000), steps=5, solver_method="greedy"):
    """
    Pre-samples the parameter space to find min/max for N and S for normalization.
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
    Qs = np.linspace(Q_range[0], Q_range[1], steps)
    n_values, s_values = [], []
    
    print(f"Pre-sampling {steps}x{steps} grid for normalization...")
    for a in alphas:
        for q in Qs:
            res = solve_cflp(w, t, r, a, q, Q0=100000, method=solver_method)
            n_open = np.sum(res["y"])
            opening_cost = float(a * (r * np.sqrt(q / 100000)) @ res["y"])
            service_cost = res["objective"] - opening_cost
            n_values.append(n_open)
            s_values.append(service_cost)
            
    return (min(n_values), max(n_values)), (min(s_values), max(s_values))

def knee_point_loss(alpha, Q, w, t, r, Q0=100000, n_norm=(0, 1), s_norm=(0, 1), solver_method="greedy"):
    """Computes the Knee-Point Loss: L(alpha, Q) = N_hat + S_hat"""
    res = solve_cflp(w, t, r, alpha, Q, Q0, method=solver_method)
    n_open = np.sum(res["y"])
    opening_cost = float(alpha * (r * np.sqrt(Q / Q0)) @ res["y"])
    service_cost = res["objective"] - opening_cost
    
    n_min, n_max = n_norm
    s_min, s_max = s_norm
    n_hat = (n_open - n_min) / (n_max - n_min) if n_max > n_min else 0.0
    s_hat = (service_cost - s_min) / (s_max - s_min) if s_max > s_min else 0.0
    
    return n_hat + s_hat, {"n_open": int(n_open), "service_cost": float(service_cost), "n_hat": float(n_hat), "s_hat": float(s_hat)}

# ── 1D SEARCH (alpha only) ──────────────────────────────────────────────────

def run_1d_search(w, t, r, Q_fixed=200000, method="golden", alpha_range=(0.1, 10.0), 
                  n_norm=(0, 1), s_norm=(0, 1), tol=1e-3, solver_method="greedy"):
    """Performs 1D search for optimal alpha."""
    history = []
    
    def f(alpha):
        loss, info = knee_point_loss(alpha, Q_fixed, w, t, r, n_norm=n_norm, s_norm=s_norm, solver_method=solver_method)
        item = {"iter": len(history), "alpha": float(alpha), "L": float(loss), **info}
        history.append(item)
        return loss

    a, b = alpha_range
    if method == "golden":
        phi = (np.sqrt(5) - 1) / 2
        x1, x2 = b - phi * (b - a), a + phi * (b - a)
        f1, f2 = f(x1), f(x2)
        
        while (b - a) > tol:
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = b - phi * (b - a)
                f1 = f(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + phi * (b - a)
                f2 = f(x2)
        alpha_opt = (a + b) / 2

    elif method == "fibonacci":
        fibs = [1, 1]
        while fibs[-1] < (b - a) / tol:
            fibs.append(fibs[-1] + fibs[-2])
        n = len(fibs) - 1
        
        x1 = a + (fibs[n-2] / fibs[n]) * (b - a)
        x2 = a + (fibs[n-1] / fibs[n]) * (b - a)
        f1, f2 = f(x1), f(x2)
        
        for k in range(n, 2, -1):
            if f1 < f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + (fibs[k-3] / fibs[k-1]) * (b - a)
                f1 = f(x1)
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + (fibs[k-2] / fibs[k-1]) * (b - a)
                f2 = f(x2)
        alpha_opt = (a + b) / 2

    return {"alpha_opt": float(alpha_opt), "L_opt": float(f(alpha_opt)), "history": history}

# ── 2D SEARCH (alpha and Q) ─────────────────────────────────────────────────

def run_2d_search(w, t, r, method="nelder_mead", alpha_range=(0.1, 10.0), Q_range=(50000, 500000),
                  n_norm=(0, 1), s_norm=(0, 1), tol=1e-3, max_iter=50, solver_method="greedy"):
    """Performs 2D search for optimal alpha and Q."""
    history = []
    
    def f(x):
        alpha = np.clip(x[0], alpha_range[0], alpha_range[1])
        Q = np.clip(x[1], Q_range[0], Q_range[1])
        loss, info = knee_point_loss(alpha, Q, w, t, r, n_norm=n_norm, s_norm=s_norm, solver_method=solver_method)
        history.append({"iter": len(history), "alpha": float(alpha), "Q": float(Q), "L": float(loss), **info})
        return loss

    if method == "grid":
        steps = 5
        alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
        Qs = np.linspace(Q_range[0], Q_range[1], steps)
        best_l, best_params = np.inf, (0, 0)
        for a in alphas:
            for q in Qs:
                l = f([a, q])
                if l < best_l:
                    best_l, best_params = l, (a, q)
        return {"alpha_opt": float(best_params[0]), "Q_opt": float(best_params[1]), "L_opt": float(best_l), "history": history}

    elif method == "nelder_mead":
        p0 = np.array([np.mean(alpha_range), np.mean(Q_range)])
        simplex = [p0, p0 + np.array([1.0, 0]), p0 + np.array([0, 50000.0])]
        values = [f(p) for p in simplex]
        
        for _ in range(max_iter):
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            if np.linalg.norm(simplex[0] - simplex[2]) < tol: break
            
            centroid = (simplex[0] + simplex[1]) / 2
            xr = centroid + 1.0 * (centroid - simplex[2])
            fr = f(xr)
            
            if values[0] <= fr < values[1]:
                simplex[2], values[2] = xr, fr
            elif fr < values[0]:
                xe = centroid + 2.0 * (centroid - simplex[2])
                fe = f(xe)
                simplex[2], values[2] = (xe, fe) if fe < fr else (xr, fr)
            else:
                xc = centroid + 0.5 * (centroid - simplex[2])
                fc = f(xc)
                if fc < values[2]:
                    simplex[2], values[2] = xc, fc
                else:
                    for i in range(1, 3):
                        simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                        values[i] = f(simplex[i])
        
        return {"alpha_opt": float(simplex[0][0]), "Q_opt": float(simplex[0][1]), "L_opt": float(values[0]), "history": history}

# ── MAIN ANALYSIS ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    w, t, r = load_optimization_data()
    print(f"Data loaded: {len(w)} neighborhoods.")
    
    # 1. Normalization
    n_norm, s_norm = pre_sample_ranges(w, t, r, steps=4)
    print(f"N Range: {n_norm}, S Range: {s_norm}")
    
    # 2. 1D Comparison: Golden vs Fibonacci
    print("\n--- 1D Search Comparison ---")
    res_golden = run_1d_search(w, t, r, method="golden", n_norm=n_norm, s_norm=s_norm)
    res_fib = run_1d_search(w, t, r, method="fibonacci", n_norm=n_norm, s_norm=s_norm)
    
    print(f"Golden Search: alpha={res_golden['alpha_opt']:.4f}, Calls={len(res_golden['history'])}")
    print(f"Fibonacci Search: alpha={res_fib['alpha_opt']:.4f}, Calls={len(res_fib['history'])}")
    save_results(res_golden, "search_1d_golden.json")
    save_results(res_fib, "search_1d_fibonacci.json")
    
    # 3. 2D Comparison: Nelder-Mead vs Grid
    print("\n--- 2D Search Comparison ---")
    res_nm = run_2d_search(w, t, r, method="nelder_mead", n_norm=n_norm, s_norm=s_norm, max_iter=20)
    res_grid = run_2d_search(w, t, r, method="grid", n_norm=n_norm, s_norm=s_norm)
    
    print(f"Nelder-Mead: alpha={res_nm['alpha_opt']:.4f}, Q={res_nm['Q_opt']:.0f}, Calls={len(res_nm['history'])}")
    print(f"Grid Search: alpha={res_grid['alpha_opt']:.4f}, Q={res_grid['Q_opt']:.0f}, Calls={len(res_grid['history'])}")
    save_results(res_nm, "search_2d_nelder_mead.json")
    save_results(res_grid, "search_2d_grid.json")
