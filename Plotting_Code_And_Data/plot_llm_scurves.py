import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# --- Setup R Path ---
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

import tempfile
local_cran = os.path.join(tempfile.gettempdir(), 'Rtmp0wZNNv', 'downloaded_packages').replace('\\', '/')
ro.r(f".libPaths(c('{local_cran}', .libPaths()))")
ro.r(".libPaths(c('C:/Users/95820/Documents/R/win-library/4.5', .libPaths()))")

try: ordinal = importr('ordinal')
except: pass

# --- Configuration ---
OUT_DIR = r"."
DATA_FILE = r"KeyinformationExtraction_LLMs.xlsx"
EXPERIMENTS = ["DSB", "FIP", "TPB"]
MODELS = ["DeepSeekV3.2", "Gemini3Pro", "GPT5.2", "Grok4", "Qwen3Max", "Sonnet4.5"]

def plogis(q): return 1.0 / (1.0 + np.exp(-q))

def get_expected_value(ctx, thresholds, beta):
    K = len(thresholds) + 1
    cum_probs = []
    for th in thresholds:
        cum_probs.append(plogis(th - beta * ctx))
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

def generate_plots():
    df = pd.read_excel(DATA_FILE)
    df['y'] = df['y'].astype(int).astype(str)
    
    for exp in EXPERIMENTS:
        df_exp = df[df["experiment"] == exp].copy()
        
        with conversion.localconverter(default_converter + pandas2ri.converter):
            r_df = conversion.py2rpy(df_exp)
        ro.globalenv['r_df'] = r_df
        ro.r('r_df$y <- ordered(r_df$y)')
        ro.r('r_df$model <- as.factor(r_df$model)')
        
        try:
            ro.r('''
            fit_b <- clmm(y ~ ctx + (1 | model), data=r_df)
            b_coef <- fit_b$beta[1]
            b_theta <- fit_b$Theta
            b_classes <- levels(r_df$y)
            ''')
            b_beta = ro.r('b_coef')[0]
            b_theta = list(ro.r('b_theta'))
            b_classes = list(ro.r('b_classes'))
        except Exception as e:
            print(f"Error fitting baseline for {exp}: {e}")
            continue
            
        plt.figure(figsize=(9, 7))
        base_x, base_y = get_expected_value_curve(b_theta, b_beta, b_classes)
        plt.plot(base_x, base_y, 'k--', linewidth=2.5, label='Baseline Ensemble (CLMM)')
        
        for m in MODELS:
            df_m = df_exp[df_exp["model"] == m].copy()
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
                m_beta = ro.r('m_coef')[0]
                m_theta = list(ro.r('m_theta'))
                m_classes = list(ro.r('m_classes'))
                
                # Area
                def int_func(x):
                    ey_m = sum(p * float(c) for p, c in zip(get_expected_value(x, m_theta, m_beta), m_classes))
                    ey_b = sum(p * float(c) for p, c in zip(get_expected_value(x, b_theta, b_beta), b_classes))
                    return ey_m - ey_b
                area, _ = integrate.quad(int_func, 0, 1)
                
                m_x, m_y = get_expected_value_curve(m_theta, m_beta, m_classes)
                plt.plot(m_x, m_y, label=f"{m} (Area: {area:+.3f})")
                
            except Exception as e:
                print(f"Skipping {m} in {exp} due to fit error: {e}")
                
        plt.title(f"S-Curves for Selected LLM Trials: {exp}")
        plt.xlabel("Contextual Belief (Normalized 0-1)")
        plt.ylabel("Expected Value E[Y|X] (1=Aggressive, 5=Cautious)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        out_path = os.path.join(OUT_DIR, f"{exp}_sampled_curves.png")
        plt.savefig(out_path, dpi=120)
        plt.close()
        print(f"Saved {out_path}")

if __name__ == '__main__':
    generate_plots()
