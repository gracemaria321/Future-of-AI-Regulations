
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Outputs (created under ./outputs/):
 - panel_data.csv
 - did_results.txt (if statsmodels available)
 - its_results.txt (if statsmodels available)
 - correlation_delta_innovation.csv
 - fig_timeseries_milestones.png
 - fig_heatmap_change.png
 - fig_stacked_composition_pre.png / fig_stacked_composition_post.png
 - fig_radar_<region>.png (if 6-dimension governance scores present)
 - (optional) enforcement/compliance bars if columns exist
"""

import os
import argparse
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Optional statsmodels (DiD/ITS). If not present, code will skip modeling gracefully.
HAVE_STATSMODELS = True
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
except Exception:
    HAVE_STATSMODELS = False

# ---- plotting style (safe across Matplotlib versions) ----
try:
    plt.style.use("ggplot")
except Exception:
    pass

OUT_DIR = os.path.join(os.path.dirname(__file__), "EvaluationOutputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Run AI governance vs cybercrime analysis on your CSVs")
    ap.add_argument("--cyber", type=str, required=True, help="Path to cybersecurity_large_synthesized_data.csv")
    ap.add_argument("--reg",   type=str, required=True, help="Path to ai_regulations_global_2025_worldcover_part3.csv")
    ap.add_argument("--ai",    type=str, required=True, help="Path to Global_AI_Content_Impact_Dataset.csv")
    ap.add_argument("--save-prefix", type=str, default="", help="Optional prefix added to output filenames")
    return ap.parse_args()

# ---------------- Utils ----------------
def _safe_read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"Failed reading {path}: {e}")

def _to_datetime(series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", infer_datetime_format=True)

def _first_existing(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None

# ---------------- Load & Preprocess ----------------
def load_cybercrime(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)
    # Expected columns (from your file): attack_type, timestamp, location, industry, outcome, attack_severity ...
    # Parse time
    if "timestamp" in df.columns:
        df["date"] = _to_datetime(df["timestamp"]).dt.to_period("M").dt.to_timestamp()
    elif "date" in df.columns:
        df["date"] = _to_datetime(df["date"]).dt.to_period("M").dt.to_timestamp()
    else:
        raise ValueError("Cyber dataset must have 'timestamp' or 'date' column.")

    # Region & outcome normalization
    df["region"] = df["location"].fillna("Unknown")
    df["outcome"] = df["outcome"].str.lower()
    # outcome: success/ failure → binary
    df["is_success"] = np.where(df["outcome"] == "success", 1, 0)
    # severity numeric if present
    if "attack_severity" in df.columns:
        df["attack_severity"] = pd.to_numeric(df["attack_severity"], errors="coerce")

    # Keep minimum set
    keep = ["region", "date", "attack_type", "industry", "is_success", "attack_severity"]
    for k in keep:
        if k not in df.columns:
            # Missing columns become NA
            df[k] = np.nan
    df = df[keep].dropna(subset=["region", "date"])
    return df


def load_regulations(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)

    # Normalize headers from your CSV
    df = df.rename(columns={
        "Country/Region": "region",
        "Law/Policy Name": "law_name",
        "Status": "status",
        "Year": "year",
        "Scope": "scope",
        "Provenance": "source"
    })

    # Keep only non-empty regions
    df = df.dropna(subset=["region"])
    df["region"] = df["region"].astype(str).str.strip()

    # --- Robust year handling: floats like 2024.0 -> 2024 ---
    # (many rows in your file have '2024.0')
    df["year_num"] = pd.to_numeric(df["year"], errors="coerce")
    df["year_int"] = df["year_num"].astype("Int64")

    # Build 'effective_date' from year (Jan 1 of that year). If year missing, NaT.
    df["effective_date"] = pd.to_datetime(df["year_int"], format="%Y", errors="coerce")

    # Optional: remove "Nothing found" rows
    if "status" in df.columns:
        df = df[df["status"].notna()]
        df = df[~df["status"].str.contains("Nothing found", case=False, na=False)]

    return df


def load_ai_content(path: str) -> pd.DataFrame:
    df = _safe_read_csv(path)
    # Normalize a few names
    df = df.rename(columns={
        "Country": "country",
        "Year": "year",
        "Industry": "industry",
        "AI Adoption Rate (%)": "ai_adoption_pct",
        "AI-Generated Content Volume (TBs per year)": "content_tb",
        "Job Loss Due to AI (%)": "job_loss_pct",
        "Revenue Increase Due to AI (%)": "revenue_up_pct",
        "Human-AI Collaboration Rate (%)": "collab_pct",
        "Top AI Tools Used": "top_tools",
        "Regulation Status": "regulation_status",
        "Consumer Trust in AI (%)": "trust_pct",
        "Market Share of AI Companies (%)": "ai_market_share_pct"
    })
    # Clean types
    for c in ["year", "ai_adoption_pct", "content_tb", "job_loss_pct", "revenue_up_pct",
              "collab_pct", "trust_pct", "ai_market_share_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["year"], errors="coerce", format="%Y")
    # Region naming: we’ll treat "country" as region for plotting
    df["region"] = df["country"]
    return df

# ---------------- Region Mapping (EU aggregation optional) ----------------
EU_MEMBERS_SIMPLE = {
    # Minimal working set; add more if needed
    "France", "Germany", "Spain", "Italy", "Netherlands", "Belgium", "Portugal",
    "Poland", "Austria", "Sweden", "Denmark", "Finland", "Czechia", "Ireland",
    "Greece", "Hungary", "Romania", "Bulgaria", "Croatia", "Slovakia", "Slovenia",
    "Lithuania", "Latvia", "Estonia", "Luxembourg", "Malta", "Cyprus"
}


def map_to_reg_region(cyber_df: pd.DataFrame, reg_df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach regulation flags to cyber data.
    - If reg_df has 'European Union', apply earliest effective_date to EU members present.
    - Otherwise match region names as-is.
    """
    df = cyber_df.copy()

    # Synonyms to improve matching (extend as needed)
    synonyms = {
        "USA": "United States",
        "U.S.": "United States",
        "UK": "United Kingdom",
        "EU": "European Union"
    }
    df["region"] = df["region"].astype(str).str.strip().replace(synonyms)
    reg_df = reg_df.copy()
    reg_df["region"] = reg_df["region"].astype(str).str.strip().replace(synonyms)

    # Guarantee 'effective_date' exists (in case upstream load didn't create it)
    if "effective_date" not in reg_df.columns:
        reg_df["year_num"] = pd.to_numeric(reg_df.get("year"), errors="coerce")
        reg_df["year_int"] = reg_df["year_num"].astype("Int64")
        reg_df["effective_date"] = pd.to_datetime(reg_df["year_int"], format="%Y", errors="coerce")

    # Build earliest effective date per region
    reg_dates: Dict[str, pd.Timestamp] = {}

    # EU: apply earliest EU effective date to member states (when present)
    eu_rows = reg_df[reg_df["region"].str.contains("European Union", case=False, na=False)]
    if not eu_rows.empty and eu_rows["effective_date"].notna().any():
        eu_first = eu_rows["effective_date"].dropna().min()
        for r in df["region"].unique():
            if r in EU_MEMBERS_SIMPLE:
                reg_dates[r] = eu_first

    # Other regions: earliest effective date per region (skip NaT)
    for rgn, sub in reg_df.groupby("region", dropna=True):
        first_dt = sub["effective_date"].dropna().min()
        if pd.notna(first_dt) and rgn.lower() != "european union":
            reg_dates[rgn] = min(first_dt, reg_dates.get(rgn, first_dt))

    # Flag post_reg and compute months_since_first_reg safely
    df["post_reg"] = False
    df["months_since_first_reg"] = np.nan

    def months_diff_safe(d: pd.Timestamp, ref: pd.Timestamp) -> float:
        if pd.isna(d) or pd.isna(ref):
            return np.nan
        return (d.year - ref.year) * 12 + (d.month - ref.month)

    for r, first_dt in reg_dates.items():
        mask = (df["region"] == r)
        df.loc[mask, "post_reg"] = df.loc[mask, "date"] >= pd.Timestamp(first_dt)
        df.loc[mask, "months_since_first_reg"] = df.loc[mask, "date"].apply(
            lambda d: months_diff_safe(d, pd.Timestamp(first_dt))
        )

    return df


# ---------------- Panel + Modeling ----------------
def build_panel(cyber_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate to region-month-attack_type level:
     - incidents (count)
     - success_rate
     - severity_avg
    Outcome variable y := incidents per month (primary).
    """
    grp = cyber_df.groupby(["region", "date", "attack_type"], as_index=False).agg(
        incidents=("is_success", "size"),
        success_rate=("is_success", "mean"),
        severity_avg=("attack_severity", "mean"),
        post_reg=("post_reg", "max")
    )
    grp["y"] = grp["incidents"]

    # Treated: regions that ever have post_reg==True
    treated_regions = grp.loc[grp["post_reg"] == True, "region"].unique().tolist()
    grp["treated"] = grp["region"].isin(treated_regions).astype(int)

    # Add year/month for FE and plotting
    grp["year"] = grp["date"].dt.year
    grp["month"] = grp["date"].dt.month
    return grp

def run_did(panel: pd.DataFrame, save_path: str) -> None:
    if not HAVE_STATSMODELS:
        print("[INFO] statsmodels not available; skipping DiD.")
        return
    data = panel.copy()
    data["time"] = data["date"].dt.to_period("M").astype(str)
    model = ols("y ~ treated*post_reg + C(region) + C(time)", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["region"]}
    )
    with open(save_path, "w") as f:
        f.write(model.summary().as_text())
    print("[DONE] DiD results saved:", save_path)

def run_its(cyber_df: pd.DataFrame, panel: pd.DataFrame, save_path: str) -> None:
    if not HAVE_STATSMODELS:
        print("[INFO] statsmodels not available; skipping ITS.")
        return
    merged = panel.merge(
        cyber_df[["region", "date", "months_since_first_reg"]].drop_duplicates(),
        on=["region", "date"], how="left"
    )
    data = merged.dropna(subset=["months_since_first_reg"]).copy()
    data["t"] = data.groupby("region")["date"].rank(method="first")
    data["post"] = data["post_reg"].astype(int)
    data["t_after_post"] = data["post"] * data["months_since_first_reg"].clip(lower=0)

    model = ols("y ~ t + post + t_after_post + C(region)", data=data).fit(
        cov_type="cluster", cov_kwds={"groups": data["region"]}
    )
    with open(save_path, "w") as f:
        f.write(model.summary().as_text())
    print("[DONE] ITS results saved:", save_path)

# ---------------- Innovation / Trust metrics ----------------
def build_innovation_trust(ai_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize innovation/trust by region-year (ai_adoption_pct, content_tb, revenue_up_pct, trust_pct).
    Used to correlate with post-pre crime deltas.
    """
    cols = ["ai_adoption_pct", "content_tb", "revenue_up_pct", "trust_pct"]
    present = [c for c in cols if c in ai_df.columns]
    if not present:
        return pd.DataFrame()

    agg = ai_df.groupby(["region", "date"], as_index=False)[present].mean()
    return agg

# ---------------- Plots ----------------
def plot_time_series(panel: pd.DataFrame, save_path: str, milestones: Dict[str, pd.Timestamp] = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    ag = panel.groupby(["region", "date"])["y"].sum().reset_index()
    for r in sorted(ag["region"].unique()):
        sub = ag[ag["region"] == r]
        ax.plot(sub["date"], sub["y"], label=r, linewidth=1.5)
    if milestones:
        for label, dt in milestones.items():
            ax.axvline(pd.Timestamp(dt), color="red", linestyle="--", alpha=0.6)
            ax.text(pd.Timestamp(dt), ax.get_ylim()[1]*0.95, label, rotation=90,
                    va="top", ha="right", fontsize=8, color="red")
    ax.set_title("Monthly Cybercrime Incidents by Region with Regulation Milestones")
    ax.set_xlabel("Date")
    ax.set_ylabel("Incidents (count)")
    ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    print("[DONE] Plot saved:", save_path)

def plot_heatmap_change(panel: pd.DataFrame, save_path: str):
    pre = panel[panel["post_reg"] == False].groupby(["region", "attack_type"])["y"].mean().unstack("attack_type")
    post = panel[panel["post_reg"] == True].groupby(["region", "attack_type"])["y"].mean().unstack("attack_type")
    change = (post - pre).fillna(0)

    fig, ax = plt.subplots(figsize=(10, 6))
    # Simple heatmap using imshow (no seaborn needed)
    data = change.values
    im = ax.imshow(data, cmap="coolwarm", aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
    ax.set_xticks(range(change.shape[1])); ax.set_xticklabels(change.columns, rotation=45, ha="right")
    ax.set_yticks(range(change.shape[0])); ax.set_yticklabels(change.index)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Post − Pre (avg monthly incidents)")
    # Add annotations
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Change in Incident Rate by Region and Attack Type (Post − Pre)")
    plt.tight_layout()
    fig.savefig(save_path, dpi=200)
    print("[DONE] Plot saved:", save_path)

def plot_stacked_composition(panel: pd.DataFrame, save_path_pre: str, save_path_post: str, focus_regions=None):
    focus_regions = focus_regions or ["EU", "USA", "UK", "Germany", "France"]
    # Sum by attack_type
    pre = panel[(panel["post_reg"] == False) & (panel["region"].isin(focus_regions))] \
        .groupby(["region", "attack_type"])["y"].sum().unstack("attack_type").fillna(0)
    post = panel[(panel["post_reg"] == True) & (panel["region"].isin(focus_regions))] \
        .groupby(["region", "attack_type"])["y"].sum().unstack("attack_type").fillna(0)

    def _plot(mat: pd.DataFrame, title: str, path: str):
        mat_pct = mat.divide(mat.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
        fig, ax = plt.subplots(figsize=(12, 6))
        bottom = np.zeros(len(mat_pct.index))
        for atk in mat_pct.columns:
            ax.bar(mat_pct.index, mat_pct[atk].values, bottom=bottom, label=atk)
            bottom += mat_pct[atk].values
        ax.set_title(title); ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
        plt.tight_layout(); fig.savefig(path, dpi=200); print("[DONE] Plot saved:", path)

    _plot(pre, "Attack Composition (%): Pre‑Regulation", save_path_pre)
    _plot(post, "Attack Composition (%): Post‑Regulation", save_path_post)

def plot_radar(govern_df: pd.DataFrame):
    """
    Radar per region if governance_df contains the six dimensions:
    ['Compliance & Enforcement','Impact on Innovation','Safety & Trustworthiness',
     'Transparency & Accountability','Ethical Considerations','Societal Impact']
    """
    if govern_df.empty:
        print("[INFO] Governance radar skipped: no 6-dimension scores present.")
        return

    dims = [
        "Compliance & Enforcement",
        "Impact on Innovation",
        "Safety & Trustworthiness",
        "Transparency & Accountability",
        "Ethical Considerations",
        "Societal Impact"
    ]
    # Check presence
    if not set(dims).issubset(set(govern_df["dimension"].unique())):
        print("[INFO] Governance radar skipped: missing one or more dimensions.")
        return

    piv = govern_df.pivot_table(index="region", columns="dimension", values="score", aggfunc="mean")
    angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])

    for r in piv.index:
        scores = piv.loc[r, dims].fillna(piv[dims].mean()).values
        scores = np.concatenate([scores, [scores[0]]])

        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(angles, scores, "o-", linewidth=2, label=r)
        ax.fill(angles, scores, alpha=0.25)
        ax.set_thetagrids(angles[:-1]*180/np.pi, dims)
        ax.set_title(f"Governance Evaluation Radar: {r}")
        ax.legend(loc="upper right")
        fn = os.path.join(OUT_DIR, f"fig_radar_{r.replace(' ', '_')}.png")
        fig.savefig(fn, dpi=200)
        print("[DONE] Plot saved:", fn)

# ---------------- Main ----------------
def main():
    args = parse_args()
    cyber = load_cybercrime(args.cyber)
    regs  = load_regulations(args.reg)
    ai    = load_ai_content(args.ai)

    # Attach regulation flags (including EU aggregation if present)
    cyber = map_to_reg_region(cyber, regs)

    # Build panel
    panel = build_panel(cyber)
    panel_fn = os.path.join(OUT_DIR, f"{args.save_prefix}panel_data.csv")
    panel.to_csv(panel_fn, index=False); print("[DONE] Panel saved:", panel_fn)

    # Modeling (optional)
    did_fn = os.path.join(OUT_DIR, f"{args.save_prefix}did_results.txt")
    its_fn = os.path.join(OUT_DIR, f"{args.save_prefix}its_results.txt")
    run_did(panel, did_fn)
    run_its(cyber, panel, its_fn)

    # Innovation/trust summary and correlation with post-pre deltas
    innov = build_innovation_trust(ai)
    if not innov.empty:
        # Align by region-year (use average across post/pre windows)
        post_avg = panel[panel["post_reg"] == True].groupby("region")["y"].mean()
        pre_avg  = panel[panel["post_reg"] == False].groupby("region")["y"].mean()
        delta = (post_avg - pre_avg).reset_index().rename(columns={"y": "delta_post_minus_pre"})

        # Reduce innov to region-level mean
        innov_reg = innov.groupby("region", as_index=False)[["ai_adoption_pct","content_tb","revenue_up_pct","trust_pct"]].mean()
        corr_df = delta.merge(innov_reg, on="region", how="left")
        corr_df.to_csv(os.path.join(OUT_DIR, f"{args.save_prefix}correlation_delta_innovation.csv"), index=False)
        for col in ["ai_adoption_pct","content_tb","revenue_up_pct","trust_pct"]:
            if col in corr_df.columns:
                c = corr_df["delta_post_minus_pre"].corr(corr_df[col])
                print(f"[INFO] Correlation delta vs {col}: {c:.3f}")
    else:
        print("[INFO] Innovation/trust dataset has no numeric columns for correlation; skipped.")

    # Milestones (EU AI Act phases to annotate, if 'European Union' present)
    # milestones = {}
    # eu_row = regs[regs["region"].str.contains("European Union", case=False, na=False)]
    # if not eu_row.empty and eu_row["effective_date"].notna().any():
    #     # Use the provided year as first milestone; you can add phase dates here if you want.
    #     milestones["EU AI Act (published)"] = eu_row["effective_date"].dropna().min()

    
    milestones = {
        "EU AI Act (published)": pd.Timestamp("2024-01-01"),  # or from your CSV
        "NIST AI RMF (updated)": pd.Timestamp("2024-07-01"),
        "YUVAi Program (implemtation)": pd.Timestamp("2023-10-01")
    }

    plot_time_series(panel, os.path.join(OUT_DIR, f"{args.save_prefix}fig_timeseries_milestones.png"), milestones=milestones)
    plot_heatmap_change(panel, os.path.join(OUT_DIR, f"{args.save_prefix}fig_heatmap_change.png"))
    plot_stacked_composition(panel,
                             os.path.join(OUT_DIR, f"{args.save_prefix}fig_stacked_composition_pre.png"),
                             os.path.join(OUT_DIR, f"{args.save_prefix}fig_stacked_composition_post.png"),
                             focus_regions=["France","Germany","UK","USA","China","Canada","Australia","India"])

    # Governance radar — requires a separate dataset with the 6 dimensions (not included in your CSVs).
    # If you later add such a CSV with columns: region, dimension, score, you can load and pass it here.
    # Example:
    # govern_df = pd.read_csv("governance_metrics_6_dimensions.csv")
    # plot_radar(govern_df)

    print("[DONE] All outputs written to:", OUT_DIR)

if __name__ == "__main__":
    main()
