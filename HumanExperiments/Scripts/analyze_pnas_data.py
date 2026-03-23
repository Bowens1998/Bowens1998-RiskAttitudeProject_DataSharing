import pandas as pd
import json
import uuid
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_script_dir, 'PNAS Human Experiment.xlsx')
xl = pd.ExcelFile(file_path)

dfs = {}
for sheet in ['TPB', 'FIP', 'DSB']:
    df = xl.parse(sheet)
    # Filter the TestByBowen row
    df = df[df['Subject ID'] != 'TestByBowen'].reset_index(drop=True)
    dfs[sheet] = df

# Identify the rows where Subject ID is NaN (first 11 valid rows)
# Wait, let's just create 11 uuids.
num_missing = dfs['TPB']['Subject ID'].isna().sum()
print(f"Number of rows missing Subject ID: {num_missing}")

generated_uuids = [str(uuid.uuid4()) for _ in range(num_missing)]

for sheet in ['TPB', 'FIP', 'DSB']:
    # Replace NaN with generated UUIDs in order
    mask = dfs[sheet]['Subject ID'].isna()
    dfs[sheet].loc[mask, 'Subject ID'] = generated_uuids

print("Identified and assigned UUIDs for missing Subject IDs.")

def is_usable_tpb(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, f"Expected 5 trials, got {len(trials)}"
        finalized_answers = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, f"Trial ended with {t.get('end_reason')}"
            finalized_answers.append(str(t.get('finalized', '')))
        
        if len(set(finalized_answers)) == 1:
            return False, "All 5 trials have identical finalized diagnosis"
            
        return True, "Usable"
    except Exception as e:
        return False, f"Error parsing: {str(e)}"

def is_usable_fip(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, f"Expected 5 trials, got {len(trials)}"
        state_finals = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, f"Trial ended with {t.get('end_reason')}"
            # Extract sum of absolute allocations or just the string representation
            sf = t.get('state_final', {})
            state_finals.append(str(sf))
            
        if len(set(state_finals)) == 1:
            return False, "All 5 trials have identical risk allocations"
            
        return True, "Usable"
    except Exception as e:
        return False, f"Error parsing: {str(e)}"

def is_usable_dsb(json_data):
    try:
        data = json.loads(json_data)
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False, f"Expected 5 trials, got {len(trials)}"
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False, f"Trial ended with {t.get('end_reason')}"
        return True, "Usable"
    except Exception as e:
        return False, f"Error parsing: {str(e)}"

results = []

n_total = len(dfs['TPB'])
print(f"Total entries (excluding TestByBowen): {n_total}")

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

print("\n--- Data Quality Report ---")
print(f"Total Subjects: {n_total}")
print(f"Usable in TPB: {results_df['TPB Usable'].sum()}")
print(f"Usable in FIP: {results_df['FIP Usable'].sum()}")
print(f"Usable in DSB: {results_df['DSB Usable'].sum()}")
print(f"Fully Usable (across all 3): {results_df['All Usable'].sum()}")

print("\nDetailed Issues:")
for _, row in results_df[~results_df['All Usable']].iterrows():
    print(f"Subject {row['Subject ID']}:")
    if not row['TPB Usable']: print(f"  TPB: {row['TPB Reason']}")
    if not row['FIP Usable']: print(f"  FIP: {row['FIP Reason']}")
    if not row['DSB Usable']: print(f"  DSB: {row['DSB Reason']}")

# Save the cleaned and filled data to a new Excel file
out_path = os.path.join(_script_dir, 'PNAS_Human_Experiment_Cleaned.xlsx')
with pd.ExcelWriter(out_path) as writer:
    for sheet in ['TPB', 'FIP', 'DSB']:
        dfs[sheet].to_excel(writer, sheet_name=sheet, index=False)
    results_df.to_excel(writer, sheet_name='Quality_Report', index=False)

print(f"\nCleaned data with SubjectIDs and quality report saved to: {out_path}")
