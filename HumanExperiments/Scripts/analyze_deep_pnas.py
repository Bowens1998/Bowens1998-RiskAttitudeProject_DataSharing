import pandas as pd
import json
import matplotlib.pyplot as plt
import os
import uuid

_script_dir = os.path.dirname(os.path.abspath(__file__))
# Read from the ORIGINAL file to avoid truncated JSON strings!
file_path = os.path.join(_script_dir, 'PNAS Human Experiment.xlsx')
xl = pd.ExcelFile(file_path)

dfs = {}
for sheet in ['TPB', 'FIP', 'DSB']:
    df = xl.parse(sheet)
    # Filter the TestByBowen row
    df = df[df['Subject ID'] != 'TestByBowen'].reset_index(drop=True)
    dfs[sheet] = df

# Handle missing UUIDs just like the first time
num_missing = dfs['TPB']['Subject ID'].isna().sum()
generated_uuids = [str(uuid.uuid4()) for _ in range(num_missing)]

for sheet in ['TPB', 'FIP', 'DSB']:
    mask = dfs[sheet]['Subject ID'].isna()
    dfs[sheet].loc[mask, 'Subject ID'] = generated_uuids

def is_usable_tpb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, "Incomplete Trials"
        
        finalized_answers = []
        contextual_beliefs = []
        total_steps = 0
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, "Timeout/Idle"
                
            fin = t.get('finalized', {})
            ctx_updates_count = 0
            if isinstance(fin, dict):
                finalized_answers.append(str(fin.get('ESI', '')))
                ctx_updates_count = fin.get('ctxUpdates', 0)
            else:
                finalized_answers.append(str(fin))
            
            steps_list = t.get('steps', [])
            steps_len = len(steps_list)
            ctx = t.get('ctxUpdates', 0)
            ctx_len = len(ctx) if isinstance(ctx, list) else int(ctx)
            total_steps += (steps_len + ctx_len)
            
            # Extract final ctx per trial ONLY if the user actually interacted with the slider
            if ctx_updates_count > 0:
                last_ctx = None
                for step in steps_list:
                    for upd in step.get('BC_updates', []):
                        if 'ctx' in upd:
                            last_ctx = upd['ctx']
                contextual_beliefs.append(str(last_ctx) if last_ctx is not None else 'skipped')
            else:
                contextual_beliefs.append('skipped')
            
        if len(set(finalized_answers)) == 1:
            return False, "Identical Diagnoses"
            
        # Check if all contextual beliefs are identical AND NOT skipped
        if len(set(contextual_beliefs)) == 1 and list(contextual_beliefs)[0] != 'skipped':
            return False, "Identical Contextual Beliefs"
            
        return True, "Usable"
    except Exception as e:
        return False, f"Parse Error: {str(e)[:50]}"

def is_usable_fip(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, "Incomplete Trials"
        state_finals = []
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, "Timeout/Idle"
            state_finals.append(str(t.get('state_final', {})))
            contextual_beliefs.append(str(t.get('report', {}).get('contextual', {}).get('risk')))
            
        if len(set(state_finals)) == 1:
            return False, "Identical Allocations"
            
        if len(set(contextual_beliefs)) == 1:
            return False, "Identical Contextual Beliefs"
            
        return True, "Usable"
    except Exception as e:
        return False, f"Parse Error: {str(e)[:50]}"

def is_usable_dsb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, "Incomplete Trials"
            
        total_steps = 0
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, "Timeout/Idle"
            total_steps += len(t.get('steps', []))
            contextual_beliefs.append(str(t.get('context_belief')))
            
        avg_steps = total_steps / 5.0
        # Less than 5 moves across the maze implies no real effort
        if avg_steps < 5.0:
            return False, "Minimal Movement"
            
        if len(set(contextual_beliefs)) == 1:
            return False, "Identical Contextual Beliefs"
            
        return True, "Usable"
    except Exception as e:
        return False, f"Parse Error: {str(e)[:50]}"

results = []
n_total = len(dfs['TPB'])

for idx in range(n_total):
    sub_id = dfs['TPB'].loc[idx, 'Subject ID']
    
    tpb_usable, tpb_reason = is_usable_tpb(dfs['TPB'].loc[idx, 'Data (JSON)'])
    fip_usable, fip_reason = is_usable_fip(dfs['FIP'].loc[idx, 'Data (JSON)'])
    dsb_usable, dsb_reason = is_usable_dsb(dfs['DSB'].loc[idx, 'Data (JSON)'])
    
    all_usable = tpb_usable and fip_usable and dsb_usable
    
    results.append({
        'Subject ID': sub_id,
        'TPB Usable': tpb_usable,
        'TPB Reason': tpb_reason,
        'FIP Usable': fip_usable,
        'FIP Reason': fip_reason,
        'DSB Usable': dsb_usable,
        'DSB Reason': dsb_reason,
        'All Usable': all_usable
    })

results_df = pd.DataFrame(results)

# Generate Data Quality Summary dataframe (Saving only the report, not JSON)
out_path = os.path.join(_script_dir, 'PNAS_Deep_Cleaned_Report.xlsx')
with pd.ExcelWriter(out_path) as writer:
    results_df.to_excel(writer, sheet_name='Quality_Report', index=False)

# Data for Pie Charts
plot_dir = _script_dir

def create_pie_chart(usable_counts, reason_counts, title, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Chart 1: Usable vs Unusable
    labels1 = ['Usable', 'Unusable']
    sizes1 = [usable_counts.get(True, 0), usable_counts.get(False, 0)]
    colors1 = ['#4CAF50', '#F44336']
    
    if sum(sizes1) > 0:
        ax1.pie(sizes1, labels=labels1, colors=colors1, autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
    ax1.set_title(f"{title} - Overall Usability")
    
    # Chart 2: Reasons for Unusable
    unusable_reasons = {k: v for k, v in reason_counts.items() if k != 'Usable'}
    if unusable_reasons:
        labels2 = list(unusable_reasons.keys())
        sizes2 = list(unusable_reasons.values())
        ax2.pie(sizes2, labels=labels2, autopct='%1.1f%%', startangle=90)
        ax2.set_title(f"{title} - Exclusion Reasons")
    else:
        ax2.text(0.5, 0.5, 'No Exclusions', horizontalalignment='center', verticalalignment='center', fontsize=14)
        ax2.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, filename))
    plt.close()

# TPB Chart
create_pie_chart(
    results_df['TPB Usable'].value_counts().to_dict(),
    results_df['TPB Reason'].value_counts().to_dict(),
    "TPB Experiment",
    "TPB_quality_pie.png"
)

# FIP Chart
create_pie_chart(
    results_df['FIP Usable'].value_counts().to_dict(),
    results_df['FIP Reason'].value_counts().to_dict(),
    "FIP Experiment",
    "FIP_quality_pie.png"
)

# DSB Chart
create_pie_chart(
    results_df['DSB Usable'].value_counts().to_dict(),
    results_df['DSB Reason'].value_counts().to_dict(),
    "DSB Experiment",
    "DSB_quality_pie.png"
)

# Overall usablity
overall_usable = results_df['All Usable'].value_counts().to_dict()
plt.figure(figsize=(7, 6))
plt.pie([overall_usable.get(True, 0), overall_usable.get(False, 0)], 
        labels=['Fully Usable in All 3', 'Excluded in Any'],
        colors=['#2196F3', '#FF9800'], autopct='%1.1f%%', startangle=90, explode=(0, 0.1))
plt.title("Combined Experiment Usability")
plt.savefig(os.path.join(plot_dir, "Overall_quality_pie.png"))
plt.close()

print(f"Deep Analysis complete. Total subjects analyzed: {n_total}")
print(f"TPB Valid: {results_df['TPB Usable'].sum()}")
print(f"FIP Valid: {results_df['FIP Usable'].sum()}")
print(f"DSB Valid: {results_df['DSB Usable'].sum()}")
print(f"Fully Usable subjects: {overall_usable.get(True, 0)} out of {n_total}")
print(f"Pie charts saved to {plot_dir}")
