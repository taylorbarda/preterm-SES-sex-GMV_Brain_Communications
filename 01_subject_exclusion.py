import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 01 · Subject exclusion

    Merges participant metadata with FreeSurfer **aseg** and **DK aparc** volumes, then removes
    participants whose GMV or WMV is more than **3 SD** from the mean after regressing out
    eTIV, sex, birth group and age.

    | | |
    |---|---|
    | **Inputs** | metadata (.xlsx), `aseg_volume.csv`, `aparc_stats_{lh,rh}_volume_DK.csv` |
    | **Output** | `meta_aseg_aparc_subjects_excluded_correct.csv` → used by notebook 02 |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import statsmodels.api as sm

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    return bu, config, mo, pd, plt, sm


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(config, mo, pd):
    def load_aparc(file_path, id_col):
        """FreeSurfer aparc table; global columns are dropped because aseg already has them."""
        aparc = pd.read_csv(file_path, sep="\t").rename(columns={id_col: "MRT_ID"})
        return aparc.drop(columns=["BrainSegVolNotVent", "eTIV"], errors="ignore")

    data_meta = pd.read_excel(config.META_FILE)

    data_aseg = pd.read_csv(config.ASEG_FILE, sep="\t").rename(
        columns={"Measure:volume": "MRT_ID", "EstimatedTotalIntraCranialVol": "eTIV"}
    )

    df = (
        data_meta
        .merge(data_aseg, on="MRT_ID", how="inner")
        .merge(load_aparc(config.LH_APARC_FILE, "lh.aparc.volume"), on="MRT_ID", how="inner")
        .merge(load_aparc(config.RH_APARC_FILE, "rh.aparc.volume"), on="MRT_ID", how="inner")
    )

    mo.md(
        f"Metadata: **{len(data_meta)}** participants · aseg: **{len(data_aseg)}** scans · "
        f"merged: **{df.shape[0]} rows × {df.shape[1]} columns**"
    )
    return (df,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Residual-based outlier detection

    Each volume is regressed on its predictors (OLS); participants with residuals outside
    mean ± 3 SD are flagged. eTIV residuals are computed for checking only and are **not** used
    for exclusion. GMV outliers are removed first, then WMV outliers.
    """)
    return


@app.cell
def _(df, pd, sm):
    def add_residuals(data, outcome_col, predictor_cols, resid_col):
        data = data.copy()
        model = sm.OLS(data[outcome_col], sm.add_constant(data[predictor_cols])).fit()
        data[resid_col] = model.resid

        mean_resid, sd_resid = data[resid_col].mean(), data[resid_col].std()
        bounds = (mean_resid - 3 * sd_resid, mean_resid + 3 * sd_resid)

        return data, bounds

    covariates = ["sex", "blsgroup", "Age_at_scan2"]
    residual_specs = {
        "GMV_resid_for_exclusion": ("TotalGrayVol", ["eTIV"] + covariates),
        "WMV_resid_for_exclusion": ("CerebralWhiteMatterVol", ["eTIV"] + covariates),
        "eTIV_resid_for_exclusion": ("eTIV", covariates),
    }

    df_resid = df
    bounds = {}

    for _resid_col, (_outcome, _predictors) in residual_specs.items():
        df_resid, bounds[_resid_col] = add_residuals(df_resid, _outcome, _predictors, _resid_col)

    def outside(data, resid_col):
        low, high = bounds[resid_col]
        return (data[resid_col] < low) | (data[resid_col] > high)

    bounds_table = pd.DataFrame([
        {
            "Residual": _col,
            "Lower bound": low,
            "Upper bound": high,
            "Outliers": int(outside(df_resid, _col).sum()),
            "Used for exclusion": _col != "eTIV_resid_for_exclusion",
        }
        for _col, (low, high) in bounds.items()
    ])
    return bounds_table, df_resid, outside


@app.cell
def _(bounds_table, bu, mo):
    bu.show_table(mo, bounds_table, "Residual bounds (mean ± 3 SD)")
    return


@app.cell
def _(bu, config, df_resid, mo, outside):
    df_gmv_clean = df_resid[~outside(df_resid, "GMV_resid_for_exclusion")].copy()
    df_clean = df_gmv_clean[~outside(df_gmv_clean, "WMV_resid_for_exclusion")].copy()

    df_clean.to_csv(config.EXCLUDED_FILE, index=False)

    excluded = df_resid.loc[
        ~df_resid["MRT_ID"].isin(df_clean["MRT_ID"]),
        ["MRT_ID", "sex", "blsgroup", "GMV_resid_for_exclusion", "WMV_resid_for_exclusion"],
    ]

    mo.vstack([
        mo.md(
            f"Removed **{len(df_resid) - len(df_gmv_clean)}** GMV outliers, then "
            f"**{len(df_gmv_clean) - len(df_clean)}** WMV outliers. "
            f"Final sample: **{len(df_clean)}** "
            f"({(df_clean['blsgroup'] == 1).sum()} VP/VLBW, {(df_clean['blsgroup'] == 2).sum()} FT).  \n"
            f"Saved to `{config.EXCLUDED_FILE.relative_to(config.PROJECT_DIR)}`"
        ),
        bu.show_table(mo, excluded, "Excluded participants"),
    ])
    return (df_gmv_clean, df_clean)


@app.cell
def _(df_clean, df_gmv_clean, df_resid, plt):
    def residual_histograms(before, after, resid_col, axes):
        for group_value, label, color in [(1, "Preterm", "tab:blue"), (2, "Full-term", "tab:red")]:
            for ax, data in zip(axes, [before, after]):
                ax.hist(
                    data.loc[data["blsgroup"] == group_value, resid_col],
                    bins=30, alpha=0.6, label=label, color=color, edgecolor="black",
                )

        axes[0].set_title(f"{resid_col}\nBefore outlier removal")
        axes[1].set_title(f"{resid_col}\nAfter outlier removal")

        for ax in axes:
            ax.set_xlabel(resid_col)
            ax.legend()

        axes[0].set_ylabel("Frequency")

    fig_resid, _axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")
    residual_histograms(df_resid, df_gmv_clean, "GMV_resid_for_exclusion", _axes[0])
    residual_histograms(df_gmv_clean, df_clean, "WMV_resid_for_exclusion", _axes[1])
    fig_resid.tight_layout()
    fig_resid
    return


if __name__ == "__main__":
    app.run()
