import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set seaborn style to whitegrid to match the provided image
sns.set_theme(style="whitegrid")

exp_names = ['DSB', 'TPB', 'FIP']
exp_dirs = {
    'DSB': r'c:\ICML2026\ICML2026\FullCapacityModelResults\DSB Results',
    'TPB': r'c:\ICML2026\ICML2026\FullCapacityModelResults\TPB Results',
    'FIP': r'c:\ICML2026\ICML2026\FullCapacityModelResults\FIP Results'
}

models = ['DeepSeekV3.2', 'GPT5.2', 'Gemini3Pro', 'Grok4', 'Qwen3Max', 'Sonnet4.5']
TOTAL_TRIALS = 300

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
fig.suptitle('Data Validity by Model across Experiments', fontsize=14)

def is_valid_trial(t):
    # Check for direct error keys
    if t.get('error') is not None: return False
    if t.get('api_error') is not None: return False
    
    # Check for timeouts
    if t.get('timeout') is not None and str(t.get('timeout')).lower() == 'true': return False
    
    # Check for nulls in crucial fields
    if t.get('end_reason') == 'error': return False
    if t.get('end_reason') is None and 'end_reason' in t: return False
    if 'llm_reasoning' in t and t.get('llm_reasoning') is None: return False
    if 'llm_raw' in t and t.get('llm_raw') is None: return False
    if 'finalized' in t and t.get('finalized') is None: return False
    
    # Ensure inner fields aren't completely corrupted
    if 'finalized' in t and isinstance(t['finalized'], dict):
        if any(v is None for v in t['finalized'].values()): return False
        
    return True

print("Model Validity Counts (Valid/Total):")

for i, exp in enumerate(exp_names):
    d = exp_dirs[exp]
    ax = axes[i]
    
    valid_counts = []
    total_counts = []
    
    print(f"\n--- {exp} Experiment ---")
    
    for model in models:
        json_path = os.path.join(d, f"{model}.json")
        valid = 0
        total = TOTAL_TRIALS
        
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as file:
                try:
                    data = json.load(file)
                    trials = []
                    if 'trials' in data:
                        trials = data['trials']
                    elif 'groups' in data:
                        for g in data['groups']:
                            trials.extend(g.get('trials', []))
                            
                    for t in trials:
                        if is_valid_trial(t):
                            valid += 1
                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
        
        valid_counts.append(valid)
        total_counts.append(total)
        print(f"{model}: {valid}/{total}")
        
    x = np.arange(len(models))
    
    # Plot Total (grey)
    ax.bar(x, total_counts, color='lightgrey', label='Total' if i == 0 else None)
    
    # Plot Valid (green) matching the provided image style
    ax.bar(x, valid_counts, color='#458B43', label='Valid' if i == 0 else None)
    
    ax.set_title(f"{exp} Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    if i == 0:
        ax.set_ylabel('Number of Trials')
        ax.legend()

plt.tight_layout()
out_dir = r'C:\Users\95820\.gemini\antigravity\brain\b6355536-637d-42e4-9c72-1c8fb04b7608'
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'validity_plot.png')
plt.savefig(out_path)
print(f"\nPlot saved to: {out_path}")
