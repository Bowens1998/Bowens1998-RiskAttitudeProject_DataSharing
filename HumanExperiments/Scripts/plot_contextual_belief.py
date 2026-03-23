import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(_script_dir, 'PNAS Human Experiment.xlsx')
xl = pd.ExcelFile(file_path)

tpb_ctxs = []
fip_ctxs = []
dsb_ctxs = []

df_tpb = xl.parse('TPB')
df_tpb = df_tpb[df_tpb['Subject ID'] != 'TestByBowen']
for idx, row in df_tpb.iterrows():
    try:
        data = json.loads(str(row['Data (JSON)']))
        for t in data.get('trials', []):
            fin = t.get('finalized', {})
            ctx_updates_count = fin.get('ctxUpdates', 0) if isinstance(fin, dict) else 0
            if ctx_updates_count > 0:
                last_ctx = None
                for step in t.get('steps', []):
                    for upd in step.get('BC_updates', []):
                        if 'ctx' in upd:
                            last_ctx = upd['ctx']
                if last_ctx is not None:
                    try:
                        tpb_ctxs.append(float(last_ctx))
                    except:
                        pass
    except:
        pass

df_fip = xl.parse('FIP')
df_fip = df_fip[df_fip['Subject ID'] != 'TestByBowen']
for idx, row in df_fip.iterrows():
    try:
        data = json.loads(str(row['Data (JSON)']))
        for t in data.get('trials', []):
            val = t.get('report', {}).get('contextual', {}).get('risk')
            if val is not None:
                try:
                    fip_ctxs.append(float(val))
                except:
                    pass
    except:
        pass

df_dsb = xl.parse('DSB')
df_dsb = df_dsb[df_dsb['Subject ID'] != 'TestByBowen']
for idx, row in df_dsb.iterrows():
    try:
        data = json.loads(str(row['Data (JSON)']))
        for t in data.get('trials', []):
            val = t.get('context_belief')
            if val is not None:
                try:
                    dsb_ctxs.append(float(val))
                except:
                    pass
    except:
        pass

# Create figure
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.histplot(tpb_ctxs, bins=20, ax=axes[0], color='#2196F3', kde=True)
axes[0].set_title(f'TPB Contextual Belief Distribution\n(N={len(tpb_ctxs)} trials)')
axes[0].set_xlabel('Contextual Belief Value')
axes[0].set_ylabel('Frequency')

sns.histplot(fip_ctxs, bins=20, ax=axes[1], color='#F44336', kde=True)
axes[1].set_title(f'FIP Risk Belief Distribution\n(N={len(fip_ctxs)} trials)')
axes[1].set_xlabel('Contextual Risk Slider Value')

sns.histplot(dsb_ctxs, bins=20, ax=axes[2], color='#4CAF50', kde=True)
axes[2].set_title(f'DSB Environment Risk Belief Distribution\n(N={len(dsb_ctxs)} trials)')
axes[2].set_xlabel('Contextual Belief Value')

plt.tight_layout()

# Save the plot
out_file = os.path.join(_script_dir, 'Contextual_Belief_Distribution.png')

plt.savefig(out_file, dpi=300)
print(f"Plot saved successfully to {out_file}")
