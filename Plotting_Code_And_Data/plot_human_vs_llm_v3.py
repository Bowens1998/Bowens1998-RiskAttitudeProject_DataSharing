import os
import sys
import pandas as pd
import numpy as np
import scipy.integrate as integrate
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.miscmodels.ordinal_model import OrderedModel

warnings.filterwarnings("ignore")

# --- Windows R DLL Loading Fix ---
r_dir = r"C:\Program Files\R\R-4.5.2\bin\x64"
if os.path.exists(r_dir):
    os.environ["PATH"] = r_dir + ";" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try: os.add_dll_directory(r_dir)
        except: pass

import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects import conversion, default_converter

# Library path fix
import tempfile
local_cran = os.path.join(tempfile.gettempdir(), 'Rtmp0wZNNv', 'downloaded_packages').replace('\\', '/')
ro.r(f".libPaths(c('{local_cran}', .libPaths()))")
ro.r(".libPaths(c('C:/Users/95820/Documents/R/win-library/4.5', .libPaths()))")

try: ordinal = importr('ordinal')
except Exception as e:
    print(f"Failed to load R 'ordinal'. {e}")
    sys.exit(1)


DATA_FILE_LLM = r"KeyinformationExtraction_LLMs.xlsx"
OUT_DIR = r"../Analysis_Results_Human_vs_LLMs"
EXCEL_PATH = r"../HumanExperiments/HumanExperimentRawData&PreResults/RawData/PNAS_Deep_Cleaned.xlsx"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["DeepSeekV3.2", "Gemini3Pro", "GPT5.2", "Grok4", "Qwen3Max", "Sonnet4.5"]
EXPERIMENTS = ["DSB", "FIP", "TPB"]
EPS = 1e-6
import json

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

def get_expected_value_curve(thresholds, beta, beta_se, classes, num_points=200):
    x_vals = np.linspace(0, 1, num_points)
    y_vals, y_lower, y_upper = [], [], []
    for x in x_vals:
        probs = get_expected_value(x, thresholds, beta)
        ey = sum(p * float(c) for p, c in zip(probs, classes))
        y_vals.append(ey)
        
        probs_up = get_expected_value(x, thresholds, beta + 1.0 * beta_se)
        ey_up = sum(p * float(c) for p, c in zip(probs_up, classes))
        
        probs_dn = get_expected_value(x, thresholds, beta - 1.0 * beta_se)
        ey_dn = sum(p * float(c) for p, c in zip(probs_dn, classes))
        
        
        y_upper.append(max(ey_up, ey_dn))
        y_lower.append(min(ey_up, ey_dn))
    return x_vals, np.array(y_vals), np.array(y_lower), np.array(y_upper)

def get_probabilities_curves(thresholds, beta, num_classes, num_points=200):
    x_vals = np.linspace(0, 1, num_points)
    prob_matrix = []
    for x in x_vals:
        probs = get_expected_value(x, thresholds, beta)
        prob_matrix.append(probs)
    return x_vals, np.array(prob_matrix).T

def fit_v3_models(df):
    results = {}
    with conversion.localconverter(default_converter + pandas2ri.converter):
        r_df = conversion.py2rpy(df)
        
    ro.globalenv['r_df'] = r_df
    ro.r('r_df$y <- ordered(r_df$y)')
    ro.r('r_df$model <- as.factor(r_df$model)')
    
    # Baseline (Global CLMM)
    try:
        ro.r('''
        fit_baseline <- clmm(y ~ ctx + (1 | model), data=r_df)
        b_coef <- fit_baseline$beta[1]  
        b_se <- summary(fit_baseline)$coefficients["ctx", 2]
        b_theta <- fit_baseline$Theta   
        b_classes <- levels(r_df$y)
        ''')
        results['Baseline'] = {
            'beta': ro.r('b_coef')[0], 
            'beta_se': ro.r('b_se')[0],
            'theta': list(ro.r('b_theta')), 
            'classes': list(ro.r('b_classes'))
        }
    except Exception as e:
        print(f"Error fitting Baseline CLMM: {e}")
        return None

    # V3 Logic: Independent CLM for each LLM
    for m in MODELS:
        df_m = df[df["model"] == m].copy()
        if len(df_m) == 0: continue
        with conversion.localconverter(default_converter + pandas2ri.converter):
            r_df_m = conversion.py2rpy(df_m)
        ro.globalenv['r_df_m'] = r_df_m
        ro.r('r_df_m$y <- ordered(r_df_m$y)')
        try:
            ro.r('''
            fit_m <- clm(y ~ ctx, data=r_df_m)
            m_coef <- fit_m$beta[1]
            m_se <- summary(fit_m)$coefficients["ctx", 2]
            m_theta <- fit_m$Theta
            m_classes <- levels(r_df_m$y)
            ''')
            results[m] = {
                'beta': ro.r('m_coef')[0], 
                'beta_se': ro.r('m_se')[0],
                'theta': list(ro.r('m_theta')), 
                'classes': list(ro.r('m_classes'))
            }
        except:
            pass
            
    return results

def is_usable_tpb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5: return False
        finalized_answers = []
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']: return False
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
                        if 'ctx' in upd: last_ctx = upd['ctx']
                contextual_beliefs.append(str(last_ctx) if last_ctx is not None else 'skipped')
            else:
                contextual_beliefs.append('skipped')
        if len(set(finalized_answers)) == 1: return False
        if len(set(contextual_beliefs)) == 1 and list(contextual_beliefs)[0] != 'skipped': return False
        return True
    except: return False

def is_usable_fip(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5: return False
        state_finals = []
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']: return False
            state_finals.append(str(t.get('state_final', {})))
            contextual_beliefs.append(str(t.get('report', {}).get('contextual', {}).get('risk')))
        if len(set(state_finals)) == 1: return False
        if len(set(contextual_beliefs)) == 1: return False
        return True
    except: return False

def is_usable_dsb(json_data):
    try:
        data = json.loads(str(json_data))
        trials = data.get('trials', [])
        if len(trials) != 5: return False
        total_steps = 0
        contextual_beliefs = []
        for t in trials:
            if t.get('end_reason') in ['idle', 'timeout']: return False
            total_steps += len(t.get('steps', []))
            contextual_beliefs.append(str(t.get('context_belief')))
        avg_steps = total_steps / 5.0
        if avg_steps < 5.0: return False
        if len(set(contextual_beliefs)) == 1: return False
        return True
    except: return False

def extract_tpb(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_tpb(row["Data (JSON)"]): continue
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
                        if "ctx" in u: last_ctx = u["ctx"]
                if last_ctx is None: continue
                rows.append({"subject": str(sid), "context_belief": float(last_ctx)/100.0, "y": int(esi)}) 
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
            if mask.any(): centroids[k] = X[mask].mean(axis=0)
    h_means = np.array([X[labels == k, 2].mean() for k in range(5)])
    order = np.argsort(h_means)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])

def extract_fip(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_fip(row["Data (JSON)"]): continue
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
                rows.append({"subject": str(sid), "context_belief": float(ctx)/100.0, "L": L/tot, "M": M/tot, "H": H/tot})
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
    if len(centroids) < 5: centroids = np.linspace(x.min(), x.max(), 5)
    for _ in range(30):
        dist = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dist, axis=1)
        for k in range(5):
            mask = labels == k
            if mask.any(): centroids[k] = x[mask].mean()
    order = np.argsort(-centroids)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])

def extract_dsb(df):
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        if not is_usable_dsb(row["Data (JSON)"]): continue
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


def get_human_model_and_data(exp, df_raw):
    df = pd.DataFrame()
    if exp == "TPB": df = extract_tpb(df_raw)
    elif exp == "FIP": df = extract_fip(df_raw)
    elif exp == "DSB": df = extract_dsb(df_raw)
        
    if df.empty: return None, None
        
    ctx_mean = df["context_belief"].mean()
    ctx_std = df["context_belief"].std() or 1.0
    df["ctx_z"] = (df["context_belief"] - ctx_mean) / ctx_std
    df["y"] = df["y"].astype(int)
    
    mod = OrderedModel.from_formula("y ~ ctx_z", data=df, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    
    thresh = res.params[res.params.index != "ctx_z"].values
    beta = float(res.params.get("ctx_z", 0.0))
    beta_se = float(res.bse.get("ctx_z", 0.0))
    classes_r = [str(x) for x in sorted(df["y"].unique())]
    
    return {
        "beta": beta,
        "beta_se": beta_se,
        "theta": thresh.tolist(),
        "classes": classes_r,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std
    }, df

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

def plot_human_scatter(exp, human_res, df_human):
    sns.set_theme(style="whitegrid", palette="colorblind")
    plt.figure(figsize=(8, 6))
    
    x_jittered = df_human["context_belief"] + np.random.normal(0, 0.02, size=len(df_human))
    y_jittered = df_human["y"].astype(float) + np.random.normal(0, 0.1, size=len(df_human))
    
    sns.scatterplot(x=x_jittered, y=y_jittered, color='gray', alpha=0.3, s=30, label='Human Subject Trails (Jittered)')
    
    h_x, h_y, h_lb, h_ub = get_human_expected_value_curve(human_res, 200)
    plt.plot(h_x, h_y, '-', color='#D55E00', linewidth=3.5, label=f"Ordered Logit Expected Value\n(Beta: {human_res['beta']:.2f})")
    plt.fill_between(h_x, h_lb, h_ub, color='#D55E00', alpha=0.2)
    
    plt.title(f"Human Trials: Contextual Belief vs Schematic Belief - {exp}", fontsize=14, fontweight='bold')
    plt.xlabel("Contextual Belief", fontsize=12)
    plt.ylabel("Schematic Belief (1=Cautious, 5=Aggressive)", fontsize=12)
    plt.ylim(0.5, 5.5)
    plt.xlim(-0.05, 1.05)
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    out_file = os.path.join(OUT_DIR, f"{exp}_Human_Scatter.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Generated {out_file}")

def generate_report(pivot_rank, pivot_area, llm_pivot_rank=None):
    report_path = os.path.join(OUT_DIR, "v3_human_comparison_report.md")
    
    md_content = f"""# V3 Analysis: Human vs LLM Comparison (Independent CLM)

This report evaluates human participants alongside LLMs using the **V3 Independent Ordered Regression (CLM)** methodology. 
By generating scatterplots for humans, we provide a concrete visualization of the experimental variation across subjects and display their collective S-curve mathematically regressed via Ordered Logistic Models.

**AUC Integration Metric Setup:**
$$AUC_{{score}} = \int_{{0}}^{{1}} (\text{{Expected Value}}_{{Model}}(x) - \text{{Expected Value}}_{{GlobalBaseline}}(x)) dx$$
Where a negative Area signifies an entity is globally more cautious than the AI aggregate baseline.

### 1. DSB Experiment 
*Human behaviors and LLM mappings vs V3 baseline.*
#### Human Subject Distribution
![DSB Human Scatter](DSB_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![DSB V3 S-Curves Human](DSB_ABC_plot.png)

### 2. FIP Experiment 
#### Human Subject Distribution
![FIP Human Scatter](FIP_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![FIP V3 S-Curves Human](FIP_ABC_plot.png)

### 3. TPB Experiment 
#### Human Subject Distribution
![TPB Human Scatter](TPB_Human_Scatter.png)
#### Unified S-Curve Grid (Humans + LLMs)
![TPB V3 S-Curves Human](TPB_ABC_plot.png)

---

### Final Rankings (Human + 6 LLMs)

**Global Overview (1 = Most Cautious):**

"""
    sorted_ranks = pivot_rank.sort_values("Mean_Rank").reset_index()
    
    md_content += "| Rank | Entity | Mean Rank | DSB Rank | FIP Rank | TPB Rank | Std Dev |\n"
    md_content += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
    
    for idx, row in sorted_ranks.iterrows():
        md_content += f"| **{idx+1}** | {row['Model']} | {row['Mean_Rank']:.2f} | {int(row['DSB'])} | {int(row['FIP'])} | {int(row['TPB'])} | {row['Std_Rank']:.2f} |\n"

    md_content += """
---
*Note: Human ranking reflects the collective response dynamic compared symmetrically against individual LLMs.*
"""

    if llm_pivot_rank is not None:
        md_content += "\n### LLM-Only Rankings (Excluding Humans)\n\n"
        md_content += "**LLM Overview (1 = Most Cautious):**\n\n"
        md_content += "| Rank | Entity | Mean Rank | DSB Rank | FIP Rank | TPB Rank | Std Dev |\n"
        md_content += "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        sorted_llm_ranks = llm_pivot_rank.sort_values("Mean_Rank").reset_index()
        for idx, row in sorted_llm_ranks.iterrows():
            md_content += f"| **{idx+1}** | {row['Model']} | {row['Mean_Rank']:.2f} | {int(row['DSB'])} | {int(row['FIP'])} | {int(row['TPB'])} | {row['Std_Rank']:.2f} |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)

def main():
    if not os.path.exists(DATA_FILE_LLM):
        print(f"Data file not found: {DATA_FILE_LLM}")
        sys.exit(1)
        
    print(f"Loading Raw Dataset from {EXCEL_PATH}...")
    xl = pd.ExcelFile(EXCEL_PATH)
    raw_dfs = {
        "TPB": xl.parse("TPB"),
        "FIP": xl.parse("FIP"),
        "DSB": xl.parse("DSB")
    }
    for k in raw_dfs:
        raw_dfs[k] = raw_dfs[k][raw_dfs[k]["Subject ID"] != "TestByBowen"].reset_index(drop=True)

    df_llm = pd.read_excel(DATA_FILE_LLM)
    df_llm["y"] = 6 - df_llm["y"].astype(float).astype(int) # Invert LLM labels
    all_areas = []
    
    for exp in EXPERIMENTS:
        df_exp = df_llm[df_llm["experiment"] == exp].copy()
        df_exp["y"] = df_exp["y"].astype(str)
        df_exp["ctx"] = df_exp["ctx"].astype(float)
        
        # 1. Fit V3 LLM & Baseline models
        fits = fit_v3_models(df_exp)
        if fits is None or 'Baseline' not in fits:
            print(f"Skipping {exp} due to LLM fit failure.")
            continue
            
        base_beta = fits['Baseline']['beta']
        base_theta = fits['Baseline']['theta']
        base_classes = fits['Baseline']['classes']
        
        base_beta_se = fits['Baseline']['beta_se']
        
        # 2. Fit Human model and compute scatter plots
        human_res, df_human = get_human_model_and_data(exp, raw_dfs[exp])
        if human_res and df_human is not None:
            plot_human_scatter(exp, human_res, df_human)
        
        # Plotting Setup
        # We'll plot LLMs and humans onto the same chart.
        plt.figure(figsize=(10, 7))
        sns.set_theme(style="whitegrid", palette="colorblind")
        
        base_x, base_y, b_lb, b_ub = get_expected_value_curve(base_theta, base_beta, base_beta_se, base_classes, 200)
        plt.plot(base_x, base_y, 'k--', linewidth=2.5, label='Baseline (CLMM Global)')
        
        # LLMs areas & curves
        for m in MODELS:
            if m not in fits: continue
            m_beta = fits[m]['beta']
            m_beta_se = fits[m]['beta_se']
            m_theta = fits[m]['theta']
            m_classes = fits[m]['classes']
            
            def int_func_llm(x, m_t=m_theta, m_b=m_beta, m_c=m_classes, b_t=base_theta, b_b=base_beta, b_c=base_classes):
                ey_m = sum(p * float(c) for p, c in zip(get_expected_value(x, m_t, m_b), m_c))
                ey_b = sum(p * float(c) for p, c in zip(get_expected_value(x, b_t, b_b), b_c))
                return ey_m - ey_b
                
            area_m, _ = integrate.quad(int_func_llm, 0, 1)
            
            m_x, m_y, m_lb, m_ub = get_expected_value_curve(m_theta, m_beta, m_beta_se, m_classes, 200)
            p = plt.plot(m_x, m_y, label=f"{m} (Area: {area_m:+.3f})")
            
            all_areas.append({"Experiment": exp, "Model": m, "Area": area_m})
        
        # Human area & curve
        if human_res:
            h_x, h_y, h_lb, h_ub = get_human_expected_value_curve(human_res, 200)
            
            def int_func_human(x):
                z = (x - human_res["ctx_mean"]) / human_res["ctx_std"]
                probs = get_expected_value(z, human_res["theta"], human_res["beta"])
                ey_h = sum(p * float(c) for p, c in zip(probs, human_res["classes"]))
                ey_b = sum(p * float(c) for p, c in zip(get_expected_value(x, base_theta, base_beta), base_classes))
                return ey_h - ey_b
                
            area_h, _ = integrate.quad(int_func_human, 0, 1)
            
            plt.plot(h_x, h_y, '-', color='#D55E00', linewidth=3.5, label=f"Human (Area: {area_h:+.3f})")
            all_areas.append({"Experiment": exp, "Model": "Human", "Area": area_h})
            
        plt.title(f"Expected Cautiousness by Context ({exp}) - Humans vs LLMs", fontsize=14, fontweight='bold')
        plt.xlabel("Contextual Belief [0, 1]", fontsize=12)
        plt.ylabel("Expected Ordinal Value (1=Cautious, 5=Aggressive)", fontsize=12)
        plt.ylim(1.0, 5.0)
        plt.yticks([1, 2, 3, 4, 5])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        abc_out = os.path.join(OUT_DIR, f"{exp}_ABC_plot.png")
        plt.savefig(abc_out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Generated {abc_out}")

        # ----------------------------------------------------
        # Plot 2x3 Grid of LLM Scatter + S-Curve + Baseline
        # ----------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes = axes.flatten()
        fig.suptitle(f"{exp} - Model S-Curves and Scatter Distributions\n(1=Cautious, 5=Aggressive)", fontsize=16, fontweight='bold')

        for idx, m in enumerate(MODELS):
            ax = axes[idx]
            
            # Scatter raw data for this LLM
            m_df = df_exp[df_exp["model"] == m]
            if not m_df.empty:
                x_jit = m_df["ctx"].astype(float) + np.random.normal(0, 0.02, size=len(m_df))
                y_jit = m_df["y"].astype(float) + np.random.normal(0, 0.1, size=len(m_df))
                sns.scatterplot(x=x_jit, y=y_jit, ax=ax, alpha=0.3, s=20, color='gray', label='Data')

            # Plot specific LLM S-curve + CI
            if m in fits:
                m_beta = fits[m]['beta']
                m_beta_se = fits[m]['beta_se']
                m_theta = fits[m]['theta']
                m_classes = fits[m]['classes']
                m_x, m_y, m_lb, m_ub = get_expected_value_curve(m_theta, m_beta, m_beta_se, m_classes, 200)
                
                p = ax.plot(m_x, m_y, linewidth=2.5, label='Fitted Expected Value')
                ax.fill_between(m_x, m_lb, m_ub, color=p[0].get_color(), alpha=0.2)

            # Overlay Global Baseline
            ax.plot(base_x, base_y, 'k--', linewidth=2, alpha=0.7, label='Global Baseline')

            ax.set_title(m, fontsize=12, fontweight='bold')
            ax.set_ylim(0.5, 5.5)
            ax.set_xlim(-0.05, 1.05)
            ax.set_yticks([1, 2, 3, 4, 5])
            if idx in [0, 3]:
                ax.set_ylabel("Expected Value (1=Cau, 5=Agg)", fontsize=10)
            if idx >= 3:
                ax.set_xlabel("Contextual Belief", fontsize=10)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        grid_out = os.path.join(OUT_DIR, f"{exp}_LLM_Grid_Scatter_SCurve.png")
        plt.savefig(grid_out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Generated {grid_out}")

        # ----------------------------------------------------
        # Plot 2x3 Grid of LLM Category Probabilities
        # ----------------------------------------------------
        fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
        axes2 = axes2.flatten()
        fig2.suptitle(f"{exp} - Model Category Selection Probabilities\n(Predicted Probabilities for Ordinal Ranks)", fontsize=16, fontweight='bold')
        colors = sns.color_palette("viridis", 5).as_hex()

        for idx, m in enumerate(MODELS):
            ax2 = axes2[idx]
            if m in fits:
                m_beta = fits[m]['beta']
                m_theta = fits[m]['theta']
                m_classes = fits[m]['classes']
                
                # Fetch 5 probability arrays corresponding to m_classes (1, 2, 3, 4, 5 typically)
                x_vals, prob_curves = get_probabilities_curves(m_theta, m_beta, len(m_classes), 200)
                
                for k in range(len(m_classes)):
                    category_label = m_classes[k]
                    ax2.plot(x_vals, prob_curves[k], linewidth=2.5, color=colors[k % len(colors)], label=f'Rank {category_label}')

            ax2.set_title(m, fontsize=12, fontweight='bold')
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlim(-0.05, 1.05)
            if idx in [0, 3]:
                ax2.set_ylabel("Probability [0, 1]", fontsize=10)
            if idx >= 3:
                ax2.set_xlabel("Contextual Belief", fontsize=10)
            ax2.legend(loc='best', fontsize=8)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        prob_grid_out = os.path.join(OUT_DIR, f"{exp}_LLM_Grid_Probabilities.png")
        plt.savefig(prob_grid_out, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"Generated {prob_grid_out}")


    df_areas = pd.DataFrame(all_areas)
    
    if df_areas.empty:
        print("No models were successfully fitted.")
        sys.exit(1)
        
    df_areas["Rank"] = df_areas.groupby("Experiment")["Area"].rank(ascending=True).astype(int)
    
    pivot_area = df_areas.pivot(index="Model", columns="Experiment", values="Area")
    pivot_rank = df_areas.pivot(index="Model", columns="Experiment", values="Rank")
    
    pivot_area["Mean_Area"] = pivot_area.mean(axis=1)
    pivot_rank["Mean_Rank"] = pivot_rank.mean(axis=1)
    pivot_rank["Std_Rank"] = pivot_rank.std(axis=1)
    
    df_llm_areas = df_areas[df_areas["Model"] != "Human"].copy()
    df_llm_areas["Rank"] = df_llm_areas.groupby("Experiment")["Area"].rank(ascending=True).astype(int)
    llm_pivot_rank = df_llm_areas.pivot(index="Model", columns="Experiment", values="Rank")
    llm_pivot_rank["Mean_Rank"] = llm_pivot_rank.mean(axis=1)
    llm_pivot_rank["Std_Rank"] = llm_pivot_rank[EXPERIMENTS].std(axis=1)
    
    pivot_area.to_csv(os.path.join(OUT_DIR, "ABC_Areas_V3_Humans.csv"))
    pivot_rank.to_csv(os.path.join(OUT_DIR, "ABC_Ranks_V3_Humans.csv"))
    
    print("\n--- Final V3 ABC Ranking (1=Most Cautious) ---")
    print(pivot_rank.sort_values("Mean_Rank"))
    
    generate_report(pivot_rank, pivot_area, llm_pivot_rank)
    print(f"\nAll reports generated inside {OUT_DIR}")

if __name__ == "__main__":
    main()
