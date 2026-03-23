import os
import sys
import json
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
OUT_DIR = r"../InteractiveVisualization/Assets"
EXCEL_PATH = r"../HumanExperiments/HumanExperimentRawData&PreResults/RawData/PNAS_Deep_Cleaned.xlsx"
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["DeepSeekV3.2", "Gemini3Pro", "GPT5.2", "Grok4", "Qwen3Max", "Sonnet4.5"]
EXPERIMENTS = ["DSB", "FIP", "TPB"]
EPS = 1e-6

# Math/Stats Utils
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

def get_expected_value_curve(thresholds, beta, classes, num_points=200):
    x_vals = np.linspace(0, 1, num_points)
    y_vals = []
    for x in x_vals:
        probs = get_expected_value(x, thresholds, beta)
        ey = sum(p * float(c) for p, c in zip(probs, classes))
        y_vals.append(ey)
    return x_vals, np.array(y_vals)

def fit_v3_models(df):
    results = {}
    with conversion.localconverter(default_converter + pandas2ri.converter):
        r_df = conversion.py2rpy(df)
        
    ro.globalenv['r_df'] = r_df
    ro.r('r_df$y <- ordered(r_df$y)')
    ro.r('r_df$model <- as.factor(r_df$model)')
    
    try:
        ro.r('''
        fit_baseline <- clmm(y ~ ctx + (1 | model), data=r_df)
        b_coef <- fit_baseline$beta[1]  
        b_theta <- fit_baseline$Theta   
        b_classes <- levels(r_df$y)
        ''')
        results['Baseline'] = {
            'beta': ro.r('b_coef')[0], 
            'theta': list(ro.r('b_theta')), 
            'classes': list(ro.r('b_classes'))
        }
    except Exception as e:
        print(f"Error fitting Baseline: {e}")
        return None

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
            m_theta <- fit_m$Theta
            m_classes <- levels(r_df_m$y)
            ''')
            results[m] = {
                'beta': ro.r('m_coef')[0], 
                'theta': list(ro.r('m_theta')), 
                'classes': list(ro.r('m_classes'))
            }
        except: pass
    return results

# Extraction Utils
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
                # TPB mapping to 1=Cautious, 5=Aggressive
                rows.append({"subject": str(sid), "context_belief": float(last_ctx)/100.0, "ESI": int(esi)}) 
        except: pass
    
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        df_out["y"] = df_out["ESI"]
    return df_out

def _kmeans5_order_by_h_inverted(X, random_state=42):
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), 5, replace=False)]
    for _ in range(50):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        for k in range(5):
            mask = labels == k
            if mask.any(): centroids[k] = X[mask].mean(axis=0)
    h_means = np.array([X[labels == k, 2].mean() for k in range(5)])
    
    # Original ordered it as argsort(-h_means). High High-risk -> class 1 (Aggressive).
    # We want 1=Cautious, 5=Aggressive. So High High-risk -> class 5.
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
        # Using 5=Aggressive inversion natively:
        df_fip["y"] = _kmeans5_order_by_h_inverted(X)
    return df_fip

def _kmeans1d_5_inverted(x, random_state=42):
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
            
    # x = SI = log(vcr/hpr). High SI = Vertical bias (Cautious).
    # Original used: order = np.argsort(centroids).
    # This meant lowest SI -> 1, highest SI -> 5.
    # Lowest SI = Aggressive mapped to 1.
    # We want 1=Cautious, 5=Aggressive.
    # So lowest SI -> 5, highest SI -> 1.
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
        df_dsb["y"] = _kmeans1d_5_inverted(si)
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
    classes_r = [str(x) for x in sorted(df["y"].unique())]
    
    return {
        "beta": beta,
        "theta": thresh.tolist(),
        "classes": classes_r,
        "ctx_mean": ctx_mean,
        "ctx_std": ctx_std
    }, df

def get_human_expected_value_curve(human_res, num_points=200):
    x_vals = np.linspace(0, 1, num_points)
    beta = human_res["beta"]
    theta = human_res["theta"]
    classes = human_res["classes"]
    ctx_m = human_res["ctx_mean"]
    ctx_s = human_res["ctx_std"]
    
    y_vals = []
    for x in x_vals:
        z = (x - ctx_m) / ctx_s
        probs = get_expected_value(z, theta, beta)
        ey = sum(p * float(c) for p, c in zip(probs, classes))
        y_vals.append(ey)
    return x_vals, np.array(y_vals)

def plot_human_only_baseline(exp, human_res, df_human):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(9, 6))
    
    x_jittered = df_human["context_belief"] + np.random.normal(0, 0.015, size=len(df_human))
    y_jittered = df_human["y"].astype(float) + np.random.normal(0, 0.1, size=len(df_human))
    
    sns.scatterplot(x=x_jittered, y=y_jittered, color='blue', alpha=0.35, s=40, label='Valid Human Trials (Jittered)')
    
    h_x, h_y = get_human_expected_value_curve(human_res, 200)
    plt.plot(h_x, h_y, '-', color='red', linewidth=3.5, label=f"Fit (Ordered Logit)\nBeta: {human_res['beta']:.3f}")
    
    plt.title("Risk Attitude", fontsize=16, fontweight='bold')
    plt.xlabel("Contextual Belief [0, 1]", fontsize=12)
    plt.ylabel("Schematic Belief (1=Cautious, 5=Aggressive)", fontsize=12)
    plt.yticks([1, 2, 3, 4, 5])
    plt.ylim(0.5, 5.5)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc="upper left" if human_res['beta'] < 0 else "upper right")
    plt.tight_layout()
    
    out_file = os.path.join(OUT_DIR, f"Human_Cleaned_{exp}_Scatter_SCurve.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()

def plot_human_scatter_v3(exp, human_res, df_human):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 6))
    
    x_jittered = df_human["context_belief"] + np.random.normal(0, 0.02, size=len(df_human))
    y_jittered = df_human["y"].astype(float) + np.random.normal(0, 0.1, size=len(df_human))
    
    sns.scatterplot(x=x_jittered, y=y_jittered, color='blue', alpha=0.3, s=30, label='Human Subject Trials (Jittered)')
    
    h_x, h_y = get_human_expected_value_curve(human_res, 200)
    plt.plot(h_x, h_y, '-', color='red', linewidth=3.5, label=f"Ordered Logit Expected Value\n(Beta: {human_res['beta']:.2f})")
    
    plt.title("Risk Attitude", fontsize=16, fontweight='bold')
    plt.xlabel("Contextual Belief", fontsize=12)
    plt.ylabel("Schematic Belief (1=Cautious, 5=Aggressive)", fontsize=12)
    plt.yticks([1, 2, 3, 4, 5])
    plt.ylim(0.5, 5.5)
    plt.xlim(-0.05, 1.05)
    plt.legend(loc="upper left" if human_res['beta'] < 0 else "upper right")
    plt.tight_layout()
    
    out_file = os.path.join(OUT_DIR, f"{exp}_Human_Scatter.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()

def plot_abc(exp, base_theta, base_beta, base_classes, fits, human_res):
    plt.figure(figsize=(10, 7))
    sns.set_theme(style="whitegrid")
    
    base_x, base_y = get_expected_value_curve(base_theta, base_beta, base_classes, 200)
    plt.plot(base_x, base_y, 'k--', linewidth=2.5, label='Baseline (CLMM Global)')
    
    for m in MODELS:
        if m not in fits: continue
        m_beta = fits[m]['beta']
        m_theta = fits[m]['theta']
        m_classes = fits[m]['classes']
        
        m_x, m_y = get_expected_value_curve(m_theta, m_beta, m_classes, 200)
        plt.plot(m_x, m_y, label=f"{m}")
    
    if human_res:
        h_x, h_y = get_human_expected_value_curve(human_res, 200)
        plt.plot(h_x, h_y, '-', color='red', linewidth=3.5, label=f"Human")

    plt.title("Risk Attitude", fontsize=16, fontweight='bold')
    plt.xlabel("Contextual Belief [0, 1]", fontsize=12)
    plt.ylabel("Expected Ordinal Value (1=Cautious, 5=Aggressive)", fontsize=12)
    plt.ylim(0.5, 5.5)
    plt.yticks([1, 2, 3, 4, 5])
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    abc_out = os.path.join(OUT_DIR, f"{exp}_ABC_plot.png")
    plt.savefig(abc_out, dpi=150, bbox_inches='tight')
    plt.close()

def plot_grid(exp, df_exp, fits):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, m in enumerate(MODELS):
        ax = axes[i]
        df_m = df_exp[df_exp["model"] == m].copy()
        
        if len(df_m) == 0:
            ax.set_title(m, fontsize=14, fontweight='bold')
            ax.text(0.5, 3.0, "No Data", ha='center', va='center')
            continue
            
        x_jittered = df_m["ctx"].astype(float) + np.random.normal(0, 0.02, size=len(df_m))
        y_jittered = df_m["y"].astype(float) + np.random.normal(0, 0.1, size=len(df_m))
        
        sns.scatterplot(x=x_jittered, y=y_jittered, ax=ax, color='blue', alpha=0.3, s=20)
        
        if m in fits:
            m_beta = fits[m]['beta']
            m_theta = fits[m]['theta']
            m_classes = fits[m]['classes']
            
            if len(m_theta) > 0 and len(m_classes) > 1:
                m_x, m_y = get_expected_value_curve(m_theta, m_beta, m_classes, 100)
                ax.plot(m_x, m_y, 'r-', linewidth=2.5, label=f"S-Curve (Beta: {m_beta:.2f})")
                ax.legend(loc="upper left" if m_beta < 0 else "upper right")
                ax.text(0.5, 3.0, f"Beta = {m_beta:.2f}",
                        fontsize=12, color='red', alpha=0.7,
                        ha="center", va="center", bbox=dict(facecolor='white', alpha=0.5))
            else:
                ax.text(0.5, 3.0, "Beta = 0.00\n(Distribution too targeted)",
                        fontsize=10, color='red', alpha=0.7,
                        ha="center", va="center", bbox=dict(facecolor='white', alpha=0.5))
        else:
            ax.text(0.5, 3.0, "Fit Failed", ha='center', va='center')
            
        ax.set_title(m, fontsize=14, fontweight='bold')
        ax.set_xlabel("Contextual Belief", fontsize=11)
        ax.set_ylabel("Schematic Belief (1=Cautious, 5=Aggressive)", fontsize=11)
        ax.set_ylim(0.5, 5.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.5)
        
    fig.suptitle(f"Risk Attitude", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_file = os.path.join(OUT_DIR, f"{exp}_Grid_SCurves.png")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

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
    
    # Pre-transform LLM data before passing to R
    # Make 1=Cautious, 5=Aggressive
    df_llm["y"] = df_llm["y"].astype(int)
    df_llm["y"] = 6 - df_llm["y"]
    
    for exp in EXPERIMENTS:
        print(f"Processing {exp}...")
        df_exp = df_llm[df_llm["experiment"] == exp].copy()
        
        df_exp["y"] = df_exp["y"].astype(str)
        df_exp["ctx"] = df_exp["ctx"].astype(float)
        
        fits = fit_v3_models(df_exp)
        if fits is None or 'Baseline' not in fits:
            print(f"Skipping {exp} due to LLM fit failure.")
            continue
            
        base_beta = fits['Baseline']['beta']
        base_theta = fits['Baseline']['theta']
        base_classes = fits['Baseline']['classes']
        
        human_res, df_human = get_human_model_and_data(exp, raw_dfs[exp])
        
        if human_res and df_human is not None:
            plot_human_only_baseline(exp, human_res, df_human)
            plot_human_scatter_v3(exp, human_res, df_human)
            plot_abc(exp, base_theta, base_beta, base_classes, fits, human_res)
        
        plot_grid(exp, df_exp, fits)
            
    print(f"\nAll requested figures generated inside {OUT_DIR}")

if __name__ == "__main__":
    main()
