import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import ipywidgets as widgets
from IPython.display import display, HTML

# Add project root to path
sys.path.insert(0, os.path.abspath('..'))
from model.cflp import solve_cflp

import warnings
warnings.filterwarnings('ignore')

# Modern Dark Theme Setup
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#1e1e1e", "figure.facecolor": "#121212",
    "text.color": "white", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "grid.color": "#333333"
})
colors = sns.color_palette("husl", 8)

DATA_DIR = '../data/processed'
RESULTS_DIR = '../results'
OUTPUTS_DIR = '../outputs'
os.makedirs(f'{OUTPUTS_DIR}/figures', exist_ok=True)
os.makedirs(f'{OUTPUTS_DIR}/maps', exist_ok=True)

print("Loading data...")
df_n = pd.read_csv(f'{DATA_DIR}/neighborhoods.csv')
w = df_n['population'].values.astype(float)
r = df_n['rent_per_m2'].values.astype(float)
t_blended = np.load(f'{DATA_DIR}/travel_times.npy')
t_peak = np.load(f'{DATA_DIR}/travel_times_peak.npy')
t_offpeak = np.load(f'{DATA_DIR}/travel_times_offpeak.npy')

df_cbc_greedy = pd.read_csv(f'{RESULTS_DIR}/cbc_vs_greedy.csv')
df_rob = pd.read_csv(f'{RESULTS_DIR}/robustness.csv')

with open(f'{RESULTS_DIR}/search_1d_golden.json') as f: res_golden = json.load(f)
with open(f'{RESULTS_DIR}/search_1d_fibonacci.json') as f: res_fib = json.load(f)
with open(f'{RESULTS_DIR}/search_2d_nelder_mead.json') as f: res_nm = json.load(f)
with open(f'{RESULTS_DIR}/search_2d_grid.json') as f: res_grid = json.load(f)
print("Data loaded successfully!")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. CBC vs Greedy (Runtime)
sns.barplot(data=df_cbc_greedy, x="instance_size", y="runtime_s", hue="method", ax=axes[0,0], palette="viridis")
axes[0,0].set_title("CBC vs Greedy Runtime (s)", color='white', fontsize=14)
axes[0,0].set_yscale('log')

# 2. Robustness (Jaccard vs Delta)
sns.boxplot(data=df_rob, x="delta", y="jaccard", ax=axes[0,1], color=colors[1])
axes[0,1].set_title("Robustness: Jaccard Similarity under Demand Uncertainty", color='white', fontsize=14)
axes[0,1].set_ylim(0, 1.1)

# 3. 1D Convergence
iters_g = [x['iter'] for x in res_golden['history']]
L_g = [x['L'] for x in res_golden['history']]
iters_f = [x['iter'] for x in res_fib['history']]
L_f = [x['L'] for x in res_fib['history']]
axes[1,0].plot(iters_g, L_g, marker='o', label='Golden Section', color=colors[2])
axes[1,0].plot(iters_f, L_f, marker='s', label='Fibonacci', color=colors[3])
axes[1,0].set_title("1D Search Convergence L(α)", color='white', fontsize=14)
axes[1,0].legend()

# 4. 2D Search Path (Nelder Mead)
alphas = [x['alpha'] for x in res_nm['history']]
Qs = [x['Q'] for x in res_nm['history']]
axes[1,1].plot(alphas, Qs, marker='o', linestyle='-', color=colors[4])
axes[1,1].scatter(alphas[-1], Qs[-1], color='red', s=100, label='Optimum', zorder=5)
axes[1,1].set_title("Nelder-Mead 2D Search Path", color='white', fontsize=14)
axes[1,1].set_xlabel("Alpha")
axes[1,1].set_ylabel("Capacity (Q)")
axes[1,1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUTS_DIR}/figures/results_summary.png", facecolor='#121212')
plt.show()

alphas_eval = [x['alpha'] for x in res_nm['history']]
n_dcs = [x['n_open'] for x in res_nm['history']]
s_costs = [x['service_cost'] for x in res_nm['history']]

plt.figure(figsize=(10, 6))
plt.scatter(n_dcs, s_costs, c=alphas_eval, cmap='viridis', s=100, zorder=3)
plt.plot(n_dcs, s_costs, linestyle='--', color='gray', alpha=0.5)
plt.colorbar(label='Alpha')
opt_idx = np.argmin([x['L'] for x in res_nm['history']])
opt_n_open = res_nm['history'][opt_idx]['n_open']
opt_service_cost = res_nm['history'][opt_idx]['service_cost']
plt.scatter(opt_n_open, opt_service_cost, color='red', s=200, marker='*', label='Knee-Point (Optimum)', zorder=4)

plt.title("Pareto Curve: Service Cost vs Number of Open DCs", color='white', fontsize=16)
plt.xlabel("Number of Open DCs")
plt.ylabel("Total Service Cost")
plt.legend()
plt.savefig(f"{OUTPUTS_DIR}/figures/pareto_curve.png", facecolor='#121212')
plt.show()

out = widgets.Output()
from IPython.display import clear_output

def render_dashboard(*args):
    alpha = alpha_slider.value
    Q = q_slider.value
    scenario = scenario_dropdown.value
    delta = delta_slider.value
    subset_size = subset_slider.value
    
    # 1. Prepare data based on widgets
    w_sim = w[:subset_size].copy()
    if delta > 0.0:
        np.random.seed(42)
        eps = np.random.uniform(1.0 - delta, 1.0 + delta, size=subset_size)
        w_sim = w_sim * eps
        
    t_sim = t_blended[:subset_size, :subset_size]
    if scenario == 'Peak': t_sim = t_peak[:subset_size, :subset_size]
    elif scenario == 'Off-Peak': t_sim = t_offpeak[:subset_size, :subset_size]
    
    r_sim = r[:subset_size]
    
    # 2. Run greedy solver
    res = solve_cflp(w_sim, t_sim, r_sim, alpha, Q, Q0=100_000.0, method="greedy")
    
    # 3. Create Map
    m = folium.Map(location=[41.05, 28.97], zoom_start=11, tiles="CartoDB dark_matter")
    
    open_dcs = np.where(res['y'] > 0.5)[0]
    assignment = res['z']
    
    palette = sns.color_palette("husl", len(open_dcs)).as_hex()
    
    for idx, j in enumerate(open_dcs):
        folium.Marker(
            location=[df_n.iloc[j]['lat'], df_n.iloc[j]['lon']],
            popup=f"DC: {df_n.iloc[j]['name']}",
            icon=folium.Icon(color="red", icon="info-sign")
        ).add_to(m)
        
        assigned_hoods = np.where(assignment[:, j] > 0.5)[0]
        for i in assigned_hoods:
            if i != j:
                folium.PolyLine(
                    locations=[[df_n.iloc[i]['lat'], df_n.iloc[i]['lon']], 
                               [df_n.iloc[j]['lat'], df_n.iloc[j]['lon']]],
                    color=palette[idx],
                    weight=1,
                    opacity=0.6
                ).add_to(m)
                folium.CircleMarker(
                    location=[df_n.iloc[i]['lat'], df_n.iloc[i]['lon']],
                    radius=2,
                    color=palette[idx],
                    fill=True
                ).add_to(m)
    
    # Save map
    m.save(f"{OUTPUTS_DIR}/maps/interactive_map_{scenario.lower()}.html")
    
    # 4. Show summary stats
    html = f"""
    <div style='color: white; background: #1e1e1e; padding: 15px; border-radius: 8px;'>
        <h3>Simulation Results</h3>
        <ul>
            <li><b>Open DCs:</b> {len(open_dcs)}</li>
            <li><b>Total Cost:</b> {res['objective']:,.2f} TL</li>
            <li><b>Scenario:</b> {scenario}</li>
            <li><b>Alpha:</b> {alpha} | <b>Q:</b> {Q:,.0f}</li>
        </ul>
    </div>
    """
    with out:
        clear_output(wait=True)
        display(HTML(html))
        display(m)

# Widgets setup
style = {'description_width': 'initial'}
alpha_slider = widgets.FloatSlider(value=res_nm['alpha_opt'], min=100, max=10000, step=100, description='Alpha (Cost scaling):', style=style, layout=widgets.Layout(width='500px'))
q_slider = widgets.FloatSlider(value=res_nm['Q_opt'], min=10000, max=200000, step=5000, description='Q (Capacity):', style=style, layout=widgets.Layout(width='500px'))
scenario_dropdown = widgets.Dropdown(options=['Blended', 'Peak', 'Off-Peak'], value='Blended', description='Traffic Scenario:', style=style)
delta_slider = widgets.FloatSlider(value=0.0, min=0.0, max=0.30, step=0.05, description='Demand Delta (Robustness):', style=style, layout=widgets.Layout(width='500px'))
subset_slider = widgets.IntSlider(value=100, min=10, max=890, step=10, description='Candidate DCs subset:', style=style, layout=widgets.Layout(width='500px'))

alpha_slider.observe(render_dashboard, names='value')
q_slider.observe(render_dashboard, names='value')
scenario_dropdown.observe(render_dashboard, names='value')
delta_slider.observe(render_dashboard, names='value')
subset_slider.observe(render_dashboard, names='value')

# Initialize the first map
render_dashboard()

controls = widgets.VBox([alpha_slider, q_slider, scenario_dropdown, delta_slider, subset_slider])
display(widgets.HBox([controls]))
display(out)

