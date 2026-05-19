# Traffic-Aware Distribution Center Location Optimization — Istanbul

A capacitated facility location–allocation framework for optimizing distribution center (DC) placements across Istanbul's 890 neighborhoods, incorporating real traffic patterns, neighborhood-level rent costs, and demand uncertainty.

## Problem Overview

Given a set of candidate locations (neighborhoods), the model decides:

- **Which locations** to open as distribution centers (binary decision `y_j`)
- **How to assign** each neighborhood's demand to an open DC (assignment `z_ij`)

The objective minimizes a weighted combination of:

- **Opening cost** — proportional to commercial rent and a scaling parameter α
- **Service cost** — travel-time-weighted demand between neighborhoods and their assigned DCs

Subject to capacity constraints on each DC.

## Project Structure

```
├── data/
│   ├── raw/                        # Raw source files (GeoJSON, CSV)
│   ├── processed/                  # Cleaned datasets ready for modeling
│   │   ├── neighborhoods.csv       # 890 neighborhoods with lat, lon, population, rent_per_m2
│   │   ├── rents.csv               # District-level average rents (compatibility)
│   │   ├── travel_times.npy        # Blended travel time matrix (n×n)
│   │   ├── travel_times_peak.npy   # Peak-hour travel times
│   │   └── travel_times_offpeak.npy# Off-peak travel times
│   ├── prepare_data.py             # End-to-end data pipeline
│   └── data.md                     # Data documentation
│
├── model/
│   ├── cflp.py                     # CFLP solver (CBC exact + Greedy heuristic)
│   └── robustness.py               # Monte Carlo robustness simulation
│
├── optimization/
│   └── search.py                   # Outer-layer parameter search (α, Q)
│                                   #   - 1D: Golden Section, Fibonacci
│                                   #   - 2D: Nelder-Mead, Grid Search
│
├── experiments/
│   └── cbc_vs_greedy.py            # Solver benchmarking across instance sizes
│
├── tests/
│   └── test_cflp.py                # Unit tests for the CFLP solver
│
├── results/                        # Pre-computed optimization outputs (JSON/CSV)
│
├── notebook/
│   └── main.ipynb                  # Interactive analysis notebook
│
└── outputs/
    ├── figures/                    # Static plots (dark theme)
    └── maps/                       # Folium interactive maps (HTML)
```

## Data Pipeline

The `data/prepare_data.py` script handles all preprocessing:

1. **Coordinates** — Parses IBB Muhtarlık GeoJSON to extract neighborhood centroids
2. **Population** — Merges TUIK 2025 population data at the neighborhood level
3. **Rent** — Integrates Endeksa rent data with neighborhood-level granularity and district-level fallbacks
4. **Traffic** — Fetches IBB Traffic Index API data to compute peak/off-peak speed multipliers (τ)
5. **Travel Times** — Builds pairwise haversine distance matrices scaled by traffic conditions

## Solvers

| Method | Description | Use Case |
|--------|-------------|----------|
| **CBC** | Exact MILP solver via PuLP | Small-to-medium instances (n ≤ 400) |
| **Greedy** | Greedy-add heuristic with capacity-constrained assignment | Large instances, parameter sweeps |

## Parameter Optimization

The outer-layer search finds the best (α, Q) pair by minimizing a normalized loss function that balances the number of opened DCs against total service cost:

- **1D Search** (fixed Q): Golden Section and Fibonacci methods
- **2D Search**: Nelder-Mead simplex and Grid Search (10×10 grid)

### Parameter Notes

The opening cost formula `f_j = α · r̄_d(j) · √(Q/Q₀)` uses the following parameters:

| Parameter | Value | Unit | Description |
|-----------|-------|------|-------------|
| `α` | Tuned via search | m²·month | Lease-footprint scaling factor (α = T·S₀). |
| `Q₀` | 100,000 | population | Reference DC capacity for cost normalization. |
| `Q` | 200,000 (default) | population | Maximum demand a single DC can serve. |
| `r̄_d(j)` | From data | TL/m²/month | Commercial rent at candidate location j. |

> **Note on α range:** The proposal gives α=2400 as an example (S₀=200 m², T=12 months). In the outer-layer search, α is tuned over [0.1, 10.0] because Q₀=100,000 amplifies the opening cost term, effectively rescaling the α range while maintaining consistent absolute cost values.

## Robustness Analysis

Monte Carlo simulation perturbs neighborhood demands by ±δ (5%–30%) and measures solution stability via Jaccard similarity of opened DC sets across repeated trials.

## Interactive Notebook

`notebook/main.ipynb` provides:

- **Static visualizations** — CBC vs. Greedy runtime comparison, robustness boxplots, convergence plots, Pareto frontier (all in dark theme)
- **Interactive dashboard** — Real-time parameter exploration with sliders for α, Q, traffic scenario, demand perturbation δ, and DC subset size, rendered on a Folium map

> **💡 Important Note for Notebook Usage:** To view and interact with the live map dashboard, you must run the final code cell in the notebook. The map does not render automatically upon opening the file.

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# 1. Prepare data (requires internet for IBB API)
python data/prepare_data.py

# 2. Run solver benchmarks
python experiments/cbc_vs_greedy.py

# 3. Run parameter optimization (generates search_comparison.csv)
python optimization/search.py

# 4. Run robustness simulation
python model/robustness.py

# 5. Generate all static outputs (maps, figures, comparison tables)
python generate_outputs.py

# 6. Open the notebook for interactive analysis
jupyter notebook notebook/main.ipynb
```

### Running Tests

```bash
python -m pytest tests/
```

## Team

| Member | Responsibility |
|--------|---------------|
| **Alperen** | CBC solver implementation, unit tests |
| **Furkan** | Parameter search algorithms (Golden, Fibonacci, Nelder-Mead, Grid) |
| **Hasan** | Greedy heuristic, robustness simulation |
| **Semih** | Data pipeline, notebook & visualization |