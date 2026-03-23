import pandas as pd
import json
import numpy as np
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_script_dir, 'PNAS_Human_Experiment_Cleaned.xlsx')
xl = pd.ExcelFile(file_path)

dfs = {sheet: xl.parse(sheet) for sheet in ['TPB', 'FIP', 'DSB']}

# Analysis functions
def analyze_tpb(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        times = []
        steps = []
        for t in trials:
            if 'elapsed_sec' in t: times.append(t['elapsed_sec'])
            if 'steps' in t: steps.append(len(t['steps']))
        return np.mean(times) if times else 0, np.mean(steps) if steps else 0
    except:
        return 0, 0

def analyze_fip(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        times = []
        for t in trials:
            if 'elapsed_sec' in t: times.append(t['elapsed_sec'])
        return np.mean(times) if times else 0
    except:
        return 0

def analyze_dsb(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        times = []
        steps_taken = []
        for t in trials:
            if 'elapsed_sec' in t: times.append(t['elapsed_sec'])
            if 'steps' in t: steps_taken.append(len(t['steps']))
        return np.mean(times) if times else 0, np.mean(steps_taken) if steps_taken else 0
    except:
        return 0, 0

stats = []
for idx in range(len(dfs['TPB'])):
    sub_id = dfs['TPB'].loc[idx, 'Subject ID']
    t_time, t_steps = analyze_tpb(dfs['TPB'].loc[idx, 'Data (JSON)'])
    f_time = analyze_fip(dfs['FIP'].loc[idx, 'Data (JSON)'])
    d_time, d_steps = analyze_dsb(dfs['DSB'].loc[idx, 'Data (JSON)'])
    stats.append({
        'Subject ID': sub_id,
        'TPB_avg_sec': t_time,
        'TPB_avg_steps': t_steps,
        'FIP_avg_sec': f_time,
        'DSB_avg_sec': d_time,
        'DSB_avg_steps': d_steps
    })

df_stats = pd.DataFrame(stats)
pd.set_option('display.max_columns', None)
print("Descriptive Statistics for Task Interactions:")
print(df_stats.describe())

# Identifying potential outliers (e.g., extremely fast completions or zero steps)
print("\nPotential Outliers (< 10s avg completion time or 0 avg steps):")
print("TPB very fast (< 10s):", (df_stats['TPB_avg_sec'] < 10).sum())
print("FIP very fast (< 10s):", (df_stats['FIP_avg_sec'] < 10).sum())
print("DSB very fast (< 5s):", (df_stats['DSB_avg_sec'] < 5).sum())
print("DSB very few steps (< 5):", (df_stats['DSB_avg_steps'] < 5).sum())

print("\nSample outliers:")
print(df_stats[(df_stats['TPB_avg_sec'] < 10) | (df_stats['FIP_avg_sec'] < 10) | (df_stats['DSB_avg_sec'] < 5) | (df_stats['DSB_avg_steps'] < 5)].head(10))

