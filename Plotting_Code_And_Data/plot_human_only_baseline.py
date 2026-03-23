import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings

warnings.filterwarnings("ignore")

_script_dir = r"../Plotting_Code_And_Data"
EXCEL_PATH = r"../HumanExperiments/HumanExperimentRawData&PreResults/RawData/PNAS_Deep_Cleaned.xlsx"
OUT_DIR = r"../Analysis_Results_Human_vs_LLMs"
os.makedirs(OUT_DIR, exist_ok=True)


# ==========================================
# 1. Validation Logic
# ==========================================

def is_usable_tpb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False
            
        finalized_answers = []
        contextual_beliefs = []
        
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False
                
            fin = t.get('finalized', {})
            ctx_updates_count = 0
            if isinstance(fin, dict):
                finalized_answers.append(str(fin.get('ESI', '')))
                ctx_updates_count = fin.get('ctxUpdates', 0)
            else:
                finalized_answers.append(str(fin))
            
            steps_list = t.get('steps', [])
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
            return False
            
        if len(set(contextual_beliefs)) == 1 and list(contextual_beliefs)[0] != 'skipped':
            return False
            
        return True
    except:
        return False

def is_usable_fip(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False
        state_finals = []
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False
            state_finals.append(str(t.get('state_final', {})))
            contextual_beliefs.append(str(t.get('report', {}).get('contextual', {}).get('risk')))
            
        if len(set(state_finals)) == 1:
            return False
            
        if len(set(contextual_beliefs)) == 1:
            return False
            
        return True
    except:
        return False

def is_usable_dsb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5:
            return False
            
        total_steps = 0
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']:
                return False
            total_steps += len(t.get('steps', []))
            contextual_beliefs.append(str(t.get('context_belief')))
            
        avg_steps = total_steps / 5.0
        if avg_steps < 5.0:
            return False
            
        if len(set(contextual_beliefs)) == 1:
            return False
            
        return True
    except:
        return False


# ==========================================
# 2. Extraction & Clustering Functions
# ==========================================
EPS = 1e-6

def extract_tpb(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_tpb(row["Data (JSON)"]):
            continue
            
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                fin = t.get("finalized", {})
                if not isinstance(fin, dict): continue
                esi = fin.get("ESI")
                if esi is None: continue
                
                last_ctx = None
                for step in t.get("steps", []):
                    for u in step.get("BC_updates", []):
                        if "ctx" in u:
                            last_ctx = u["ctx"]
                if last_ctx is None: continue
                
                rows.append({"subject": str(sid), "context_belief": float(last_ctx)/100.0, "y": int(esi)}) 
                # TPB Mapping: ESI 1 = Cautious. ESI 5 = Aggressive.
        except: pass
    return pd.DataFrame(rows)

def _kmeans5_order_by_h(X, random_state=42):
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), 5, replace=False)]
    for _ in range(50):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        for k in range(5):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
    h_means = np.array([X[labels == k, 2].mean() for k in range(5)])
    order = np.argsort(h_means)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])

def extract_fip(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_fip(row["Data (JSON)"]):
            continue
            
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                rep = t.get("report", {}) or {}
                ctx = (rep.get("contextual") or {}).get("risk")
                alloc = rep.get("alloc", {}) or {}
                if ctx is None or alloc.get("L") is None: continue
                
                L, M, H = float(alloc.get("L", 0)), float(alloc.get("M", 0)), float(alloc.get("H", 0))
                tot = L + M + H
                if tot == 0: continue
                rows.append({
                    "subject": str(sid),
                    "context_belief": float(ctx)/100.0,
                    "L": L/tot, "M": M/tot, "H": H/tot
                })
        except: pass
        
    df_fip = pd.DataFrame(rows)
    if not df_fip.empty:
        X = df_fip[["L", "M", "H"]].values
        df_fip["y"] = _kmeans5_order_by_h(X)
    return df_fip

def _kmeans1d_5(x, random_state=42):
    rng = np.random.default_rng(random_state)
    u_vals = np.unique(x)
    centroids = np.sort(rng.choice(u_vals, size=min(5, len(u_vals)), replace=False))
    if len(centroids) < 5:
        centroids = np.linspace(x.min(), x.max(), 5)
    for _ in range(30):
        dist = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dist, axis=1)
        for k in range(5):
            mask = labels == k
            if mask.any():
                centroids[k] = x[mask].mean()
    order = np.argsort(-centroids)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])

def extract_dsb(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_dsb(row["Data (JSON)"]):
            continue
            
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                ctx = t.get("context_belief")
                steps = t.get("steps", [])
                if ctx is None or not steps: continue
                
                actions = [s.get("action", "").upper() for s in steps]
                total = len(actions)
                if total == 0: continue
                
                vcr = max((actions.count("UP") + actions.count("DOWN")) / total, EPS)
                hpr = max(actions.count("RIGHT") / total, EPS)
                si = np.log(vcr / hpr)
                
                rows.append({"subject": str(sid), "context_belief": float(ctx)/100.0, "SI": si})
        except: pass
        
    df_dsb = pd.DataFrame(rows)
    if not df_dsb.empty:
        si = df_dsb["SI"].values
        df_dsb["y"] = _kmeans1d_5(si)
    return df_dsb


# ==========================================
# 3. Fitting & Plotting Models
# ==========================================
def plogis(q):
    return 1.0 / (1.0 + np.exp(-q))

def get_expected_value(ctx, thresholds, beta):
    cum_probs = []
    for theta in thresholds:
        cum_probs.append(plogis(theta - beta * ctx))
    cum_probs.append(1.0)
    
    probs = []
    prev_c = 0.0
    for c in cum_probs:
        probs.append(max(0.0, c - prev_c))
        prev_c = c
    return probs

def get_human_expected_value_curve(human_res, num_points=200):
    x_vals = np.linspace(0, 1, num_points)
    beta = human_res["beta"]
    beta_se = human_res.get("beta_se", 0.0)
    theta = human_res["theta"]
    classes = human_res["classes"]
    ctx_m = human_res["ctx_mean"]
    ctx_s = human_res["ctx_std"]
    
    y_vals, y_lower, y_upper = [], [], []
    for x in x_vals:
        z = (x - ctx_m) / ctx_s
        probs = get_expected_value(z, theta, beta)
        ey = sum(p * float(c) for p, c in zip(probs, classes))
        y_vals.append(ey)
        
        probs_up = get_expected_value(z, theta, beta + 1.0 * beta_se)
        ey_up = sum(p * float(c) for p, c in zip(probs_up, classes))
        
        probs_dn = get_expected_value(z, theta, beta - 1.0 * beta_se)
        ey_dn = sum(p * float(c) for p, c in zip(probs_dn, classes))
        
        y_upper.append(max(ey_up, ey_dn))
        y_lower.append(min(ey_up, ey_dn))
        
    return x_vals, np.array(y_vals), np.array(y_lower), np.array(y_upper)

def run_pipeline(exp_name, df_clean):
    if df_clean.empty:
        print(f"No valid data extracted for {exp_name}")
        return
        
    print(f"\n--- Processing {exp_name} (Cleaned Valid Trials: {len(df_clean)}) ---")
    
    ctx_mean = df_clean["context_belief"].mean()
    ctx_std = df_clean["context_belief"].std() or 1.0
    df_clean["ctx_z"] = (df_clean["context_belief"] - ctx_mean) / ctx_std
    df_clean["y"] = df_clean["y"].astype(int)
    
    mod = OrderedModel.from_formula("y ~ ctx_z", data=df_clean, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    
    thresh = res.params[res.params.index != "ctx_z"].values
    beta = float(res.params.get("ctx_z", 0.0))
    beta_se = float(res.bse.get("ctx_z", 0.0))
    classes_r = [str(x) for x in sorted(df_clean["y"].unique())]
    
    human_res = {
        "beta": beta,
        "beta_se": beta_se,
        "theta": thresh.tolist(),
        "classes": classes_r,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std
    }
    
    # Render Plot
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.figure(figsize=(9, 6))
    
    # Plot real points, jittered
    x_jittered = df_clean["context_belief"] + np.random.normal(0, 0.015, size=len(df_clean))
    y_jittered = df_clean["y"].astype(float) + np.random.normal(0, 0.1, size=len(df_clean))
    
    sns.scatterplot(x=x_jittered, y=y_jittered, color='gray', alpha=0.35, s=40, label='Valid Human Trials (Jittered)')
    
    h_x, h_y, h_lb, h_ub = get_human_expected_value_curve(human_res, 200)
    plt.plot(h_x, h_y, '-', color='#D55E00', linewidth=3.5, label=f"Fit (Ordered Logit)\nBeta: {beta:.3f}")
    plt.fill_between(h_x, h_lb, h_ub, color='#D55E00', alpha=0.2, label='95% CI Band')
    
    plt.title(f"{exp_name} Human Population Dynamics (Strict PNAS Filters Included)", fontsize=14, fontweight='bold')
    plt.xlabel("Contextual Belief [0, 1]", fontsize=12)
    plt.ylabel("Schematic Belief (1=Cautious, 5=Aggressive)", fontsize=12)
    plt.ylim(0.5, 5.5)
    plt.xlim(-0.05, 1.05)
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    out_file = os.path.join(OUT_DIR, f"Human_Cleaned_{exp_name}_Scatter_SCurve.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated standalone `{exp_name}` valid population chart.")


# ==========================================
# 4. Main Function
# ==========================================
def main():
    print(f"Loading Raw Dataset from {EXCEL_PATH}...")
    xl = pd.ExcelFile(EXCEL_PATH)
    
    df_raw_tpb = xl.parse("TPB")
    df_raw_tpb = df_raw_tpb[df_raw_tpb["Subject ID"] != "TestByBowen"].reset_index(drop=True)
    
    df_raw_fip = xl.parse("FIP")
    df_raw_fip = df_raw_fip[df_raw_fip["Subject ID"] != "TestByBowen"].reset_index(drop=True)
    
    df_raw_dsb = xl.parse("DSB")
    df_raw_dsb = df_raw_dsb[df_raw_dsb["Subject ID"] != "TestByBowen"].reset_index(drop=True)
    
    print("Initiating Deep Clean and K-Means Distribution Pipeline...")
    
    df_clean_tpb = extract_tpb(df_raw_tpb)
    run_pipeline("TPB", df_clean_tpb)
    
    df_clean_fip = extract_fip(df_raw_fip)
    run_pipeline("FIP", df_clean_fip)
    
    df_clean_dsb = extract_dsb(df_raw_dsb)
    run_pipeline("DSB", df_clean_dsb)

    print(f"\nSaved all artifacts to {OUT_DIR}.")

if __name__ == "__main__":
    main()
