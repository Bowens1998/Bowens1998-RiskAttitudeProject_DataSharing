"""
Human data: Scheme B style baseline estimation for comparison with model results.
- X = context_belief (same as in Code), Y = final belief (TPB: ESI; FIP: port_cluster; DSB: SI_cluster from actions, like Code).
- Outputs: population baseline per experiment + subject random-effect SD (where available),
  and a row "Human" compatible with mixed_effects_ranking.csv for comparison.
"""
import os
import json
import math
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

_script_dir = os.path.dirname(os.path.abspath(__file__))
XLSX_PATH = os.path.join(_script_dir, "PNAS Human Experiment.xlsx")
OUT_DIR = _script_dir
N_CLUSTERS = 5
EPS = 1e-6

# Exclude test subject
def load_sheet(sheet: str) -> pd.DataFrame:
    xl = pd.ExcelFile(XLSX_PATH)
    df = xl.parse(sheet)
    df = df[df["Subject ID"].astype(str).str.strip() != "TestByBowen"].reset_index(drop=True)
    return df

# ---------- TPB: X = context_belief (last ctx), Y = ESI ----------
def extract_tpb(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                fin = t.get("finalized", {})
                if not isinstance(fin, dict):
                    continue
                esi = fin.get("ESI")
                if esi is None:
                    continue
                last_ctx = None
                for step in t.get("steps", []):
                    for u in step.get("BC_updates", []):
                        if "ctx" in u:
                            last_ctx = u["ctx"]
                if last_ctx is None:
                    continue
                try:
                    context_belief = float(last_ctx)
                except (TypeError, ValueError):
                    continue
                rows.append({"subject": str(sid), "context_belief": context_belief, "y": int(esi)})
        except Exception:
            continue
    return pd.DataFrame(rows)

# ---------- FIP: X = context_belief (report.contextual.risk), Y = port_cluster (K-Means on L,M,H) ----------
def extract_fip(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                rep = t.get("report", {}) or {}
                ctx = (rep.get("contextual") or {}).get("risk")
                alloc = rep.get("alloc", {}) or {}
                if ctx is None or alloc.get("L") is None:
                    continue
                try:
                    context_belief = float(ctx)
                except (TypeError, ValueError):
                    continue
                rows.append({
                    "subject": str(sid),
                    "context_belief": context_belief,
                    "alloc_L": int(alloc.get("L", 0)),
                    "alloc_M": int(alloc.get("M", 0)),
                    "alloc_H": int(alloc.get("H", 0)),
                })
        except Exception:
            continue
    return pd.DataFrame(rows)

# ---------- DSB: X = context_belief (trial), Y = SI then K-Means → SI_cluster (1-5), same as Code ----------
def _compute_si(steps: list) -> float | None:
    actions = [s.get("action", "").upper() for s in steps]
    total = len(actions)
    if total == 0:
        return None
    n_up = actions.count("UP")
    n_down = actions.count("DOWN")
    n_right = actions.count("RIGHT")
    vcr = max((n_up + n_down) / total, EPS)
    hpr = max(n_right / total, EPS)
    return math.log(vcr / hpr)


def extract_dsb(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        sid = row["Subject ID"]
        try:
            data = json.loads(str(row["Data (JSON)"]))
            for t in data.get("trials", []):
                ctx = t.get("context_belief")
                steps = t.get("steps", [])
                if ctx is None or not steps:
                    continue
                try:
                    context_belief = float(ctx)
                except (TypeError, ValueError):
                    continue
                si = _compute_si(steps)
                if si is None:
                    continue
                rows.append({"subject": str(sid), "context_belief": context_belief, "SI": si})
        except Exception:
            continue
    return pd.DataFrame(rows)


def run_tpb_baseline(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ctx_z"] = (df["context_belief"] - df["context_belief"].mean()) / (df["context_belief"].std() or 1)
    df["y"] = df["y"].astype(int)
    mod = OrderedModel.from_formula("y ~ ctx_z", data=df, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    thresh = res.params[res.params.index != "ctx_z"].values
    return {
        "baseline_type": "cutpoints_logit",
        "baseline_value": thresh.tolist(),
        "ctx_coef": float(res.params.get("ctx_z", np.nan)),
        "subject_RE_SD": None,
        "n_subjects": df["subject"].nunique(),
        "n_trials": len(df),
    }


def _kmeans5_order_by_h(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """K-Means 5 clusters on X, return labels 0..4 ordered by mean of last column (H)."""
    rng = np.random.default_rng(random_state)
    centroids = X[rng.choice(len(X), N_CLUSTERS, replace=False)]
    for _ in range(50):
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = np.argmin(dist, axis=1)
        for k in range(N_CLUSTERS):
            mask = labels == k
            if mask.any():
                centroids[k] = X[mask].mean(axis=0)
    h_means = np.array([X[labels == k, 2].mean() for k in range(N_CLUSTERS)])
    order = np.argsort(h_means)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])


def run_fip_baseline(df: pd.DataFrame) -> dict:
    df = df.copy()
    X = df[["alloc_L", "alloc_M", "alloc_H"]].values
    df["y"] = _kmeans5_order_by_h(X)
    df["ctx_z"] = (df["context_belief"] - df["context_belief"].mean()) / (df["context_belief"].std() or 1)
    mod = OrderedModel.from_formula("y ~ ctx_z", data=df, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    thresh = res.params[res.params.index != "ctx_z"].values
    return {
        "baseline_type": "cutpoints_logit",
        "baseline_value": thresh.tolist(),
        "ctx_coef": float(res.params.get("ctx_z", np.nan)),
        "subject_RE_SD": None,
        "n_subjects": df["subject"].nunique(),
        "n_trials": len(df),
    }


def _kmeans1d_5(x: np.ndarray, random_state: int = 42) -> np.ndarray:
    """K-Means 5 clusters on 1D array; return labels 0..4 ordered by centroid."""
    rng = np.random.default_rng(random_state)
    centroids = np.sort(rng.choice(np.unique(x), size=min(N_CLUSTERS, len(np.unique(x))), replace=False))
    if len(centroids) < N_CLUSTERS:
        centroids = np.linspace(x.min(), x.max(), N_CLUSTERS)
    for _ in range(30):
        dist = np.abs(x[:, None] - centroids[None, :])
        labels = np.argmin(dist, axis=1)
        for k in range(N_CLUSTERS):
            mask = labels == k
            if mask.any():
                centroids[k] = x[mask].mean()
    order = np.argsort(centroids)
    mapping = {old: new + 1 for new, old in enumerate(order)}
    return np.array([mapping[labels[i]] for i in range(len(labels))])


def run_dsb_baseline(df: pd.DataFrame) -> dict:
    """DSB: global K-Means on SI → SI_cluster 1-5, then ordered logit (same as Code)."""
    df = df.copy()
    si = df["SI"].values
    df["y"] = _kmeans1d_5(si)
    df["ctx_z"] = (df["context_belief"] - df["context_belief"].mean()) / (df["context_belief"].std() or 1)
    df["y"] = df["y"].astype(int)
    mod = OrderedModel.from_formula("y ~ ctx_z", data=df, distr="logit")
    res = mod.fit(method="bfgs", disp=False)
    thresh = res.params[res.params.index != "ctx_z"].values
    return {
        "baseline_type": "cutpoints_logit",
        "baseline_value": thresh.tolist(),
        "ctx_coef": float(res.params.get("ctx_z", np.nan)),
        "subject_RE_SD": None,
        "n_subjects": df["subject"].nunique(),
        "n_trials": len(df),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading human data from", XLSX_PATH)

    df_tpb = load_sheet("TPB")
    df_fip = load_sheet("FIP")
    df_dsb = load_sheet("DSB")

    tpb_long = extract_tpb(df_tpb)
    fip_long = extract_fip(df_fip)
    dsb_long = extract_dsb(df_dsb)

    print(f"TPB: {len(tpb_long)} trials, {tpb_long['subject'].nunique()} subjects")
    print(f"FIP: {len(fip_long)} trials, {fip_long['subject'].nunique()} subjects")
    print(f"DSB: {len(dsb_long)} trials, {dsb_long['subject'].nunique()} subjects")

    # Save long-format for reproducibility
    tpb_long.to_csv(os.path.join(OUT_DIR, "human_tpb_long.csv"), index=False)
    fip_long.to_csv(os.path.join(OUT_DIR, "human_fip_long.csv"), index=False)
    dsb_long.to_csv(os.path.join(OUT_DIR, "human_dsb_long.csv"), index=False)

    # Fit baseline per experiment
    res_tpb = run_tpb_baseline(tpb_long)
    res_fip = run_fip_baseline(fip_long)
    res_dsb = run_dsb_baseline(dsb_long)

    report = []
    for name, res in [("TPB", res_tpb), ("FIP", res_fip), ("DSB", res_dsb)]:
        report.append({
            "experiment": name,
            "baseline_type": res["baseline_type"],
            "baseline_value": str(res["baseline_value"]) if isinstance(res["baseline_value"], (list, np.ndarray)) else res["baseline_value"],
            "ctx_z_coef": res["ctx_coef"],
            "subject_RE_SD": res["subject_RE_SD"],
            "n_subjects": res["n_subjects"],
            "n_trials": res["n_trials"],
        })
    report_path = os.path.join(OUT_DIR, "human_baseline_scheme_b_report.csv")
    pd.DataFrame(report).to_csv(report_path, index=False)
    print(f"\nReport saved: {report_path}")

    # One row "Human" for comparison with model mixed_effects_ranking.csv
    # Human population mean offset = 0 (reference); models' ranefs are relative to population.
    human_row = {
        "model": "Human",
        "DSB_Ranef": 0.0,
        "DSB_pval": np.nan,
        "DSB_Rank": np.nan,
        "FIP_Ranef": 0.0,
        "FIP_pval": np.nan,
        "FIP_Rank": np.nan,
        "TPB_Ranef": 0.0,
        "TPB_pval": np.nan,
        "TPB_Rank": np.nan,
        "Human_DSB_RE_SD": res_dsb["subject_RE_SD"],
        "Human_FIP_RE_SD": res_fip["subject_RE_SD"],
        "Human_TPB_RE_SD": res_tpb["subject_RE_SD"],
    }
    comparison_path = os.path.join(OUT_DIR, "human_baseline_for_comparison.csv")
    pd.DataFrame([human_row]).to_csv(comparison_path, index=False)
    print(f"Comparison row (Human = reference, offset 0): {comparison_path}")

    print("\n--- Human baseline summary ---")
    print("TPB: cutpoints (logit at ctx_z=0) =", np.round(res_tpb["baseline_value"], 4))
    print("FIP: cutpoints (logit at ctx_z=0) =", np.round(res_fip["baseline_value"], 4))
    print("DSB: cutpoints (logit at ctx_z=0) =", np.round(res_dsb["baseline_value"], 4), "(ordered logit, Y=SI_cluster)")
    print("\nTo compare with models: concatenate human_baseline_for_comparison.csv with Code/RankingAnalysisV2/SchemeB_MixedEffects/mixed_effects_ranking.csv (Human = reference, model ranefs are vs population).")
    print("Detailed baseline (cutpoints/intercept, ctx coef) in human_baseline_scheme_b_report.csv.")


if __name__ == "__main__":
    main()
