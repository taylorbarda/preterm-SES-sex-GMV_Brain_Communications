import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 04 · TIV normalization and birth-group differences

    1. Regress harmonized GMV and WMV on harmonized TIV; the residuals are the
       **TIV-normalized** outcomes.
    2. ANCOVAs for birth group (whole group and sex-stratified), Fig. 1 / Supplementary Table 1.
    3. DK lobe GMV ANCOVAs with FDR correction (lobe-level analysis requested by reviewers).

    | | |
    |---|---|
    | **Input** | `NCH_cortical_subcortical_GMV_ROIs_correct.csv` (notebook 02) |
    | **Output** | `NeuroCombat_resid_global_and_lobes_correct.csv` → notebooks 05 and 06; tables and graphs in `analysis/global_and_roi_gmv_wmv/` |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.stats as stats
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    table_dir = config.GROUP_DIR / "tables"
    graph_dir = config.GROUP_DIR / "graphs"
    return bu, config, graph_dir, mo, multipletests, np, pd, plt, sm, stats, table_dir


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(bu, config, np, pd, sm):
    def add_tiv_residual(data, outcome_col, resid_col, tiv_col="TIV_harmonized"):
        model_data = data[[outcome_col, tiv_col]].dropna()
        model = sm.OLS(model_data[outcome_col], sm.add_constant(model_data[tiv_col])).fit()

        data[resid_col] = np.nan
        data.loc[model_data.index, resid_col] = model.resid

    df = pd.read_csv(config.COMBAT_FILE)
    bu.require_columns(df, bu.DK_LOBE_GMV_COLS, "in harmonized data")

    df["birth_group"] = df["blsgroup"].map(bu.BIRTHGROUP_LABELS)
    df["sex_label"] = df["sex"].map({1: "Male", 2: "Female"})

    add_tiv_residual(df, "GMV_harmonized", "GMV_resid_har")
    add_tiv_residual(df, "WMV_harmonized", "WMV_resid_har")

    # mm3 -> mL
    df["GMV_ml"] = df["GMV_harmonized"] / 1000
    df["WMV_ml"] = df["WMV_harmonized"] / 1000
    df["TIV_ml"] = df["TIV_harmonized"] / 1000
    df["GMV_resid_ml"] = df["GMV_resid_har"] / 1000
    df["WMV_resid_ml"] = df["WMV_resid_har"] / 1000

    for _col in bu.DK_LOBE_GMV_COLS:
        df[f"{_col}_ml"] = df[_col] / 1000

    df.to_csv(config.RESID_FILE, index=False)

    df_m = df[df["sex"] == 1].copy()
    df_f = df[df["sex"] == 2].copy()
    return df, df_f, df_m


@app.cell
def _(config, df, mo):
    mo.md(
        f"Saved residualized dataset ({df.shape[0]} × {df.shape[1]}) to "
        f"`{config.RESID_FILE.relative_to(config.PROJECT_DIR)}` · "
        f"males: **{(df['sex'] == 1).sum()}**, females: **{(df['sex'] == 2).sum()}**"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Distribution checks of TIV-normalized volumes

    Histograms and Q–Q plots (Methods 2.6).
    """)
    return


@app.cell
def _(df, plt, stats):
    fig_diag, _axes = plt.subplots(2, 2, figsize=(9, 7))

    for _row, (_col, _label) in enumerate([("GMV_resid_ml", "TIV-normalized GMV (mL)"), ("WMV_resid_ml", "TIV-normalized WMV (mL)")]):
        _values = df[_col].dropna()
        _axes[_row, 0].hist(_values, bins=30, color="0.7", edgecolor="black")
        _axes[_row, 0].set_xlabel(_label)
        _axes[_row, 0].set_ylabel("Frequency")
        stats.probplot(_values, dist="norm", plot=_axes[_row, 1])
        _axes[_row, 1].set_title(f"Q–Q plot: {_label}")

    fig_diag.tight_layout()
    fig_diag
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Birth-group ANCOVAs (Type II)

    Whole group: outcome ~ C(blsgroup) + sex + age · Sex-stratified: outcome ~ C(blsgroup) + age
    """)
    return


@app.cell
def _(bu, df, df_f, df_m, pd, table_dir):
    _whole = "~ C(blsgroup) + sex + Age_at_scan2"
    _strat = "~ C(blsgroup) + Age_at_scan2"

    global_specs = {
        # name: (data, formula, plot column, y label)
        "GMV_all": (df, f"GMV_resid_har {_whole}", "GMV_resid_ml", "TIV-normalized GMV (mL)"),
        "WMV_all": (df, f"WMV_resid_har {_whole}", "WMV_resid_ml", "TIV-normalized WMV (mL)"),
        "TIV_all": (df, f"TIV_harmonized {_whole}", "TIV_ml", "Harmonized TIV (mL)"),
        "GMV_males": (df_m, f"GMV_resid_har {_strat}", "GMV_resid_ml", "TIV-normalized GMV in males (mL)"),
        "GMV_females": (df_f, f"GMV_resid_har {_strat}", "GMV_resid_ml", "TIV-normalized GMV in females (mL)"),
        "WMV_males": (df_m, f"WMV_resid_har {_strat}", "WMV_resid_ml", "TIV-normalized WMV in males (mL)"),
        "WMV_females": (df_f, f"WMV_resid_har {_strat}", "WMV_resid_ml", "TIV-normalized WMV in females (mL)"),
    }

    global_results = {
        name: bu.run_ancova(data, formula) for name, (data, formula, _c, _l) in global_specs.items()
    }

    for _name, _result in global_results.items():
        _result["table"].to_csv(table_dir / f"{_name}_ANCOVA.csv")

    global_summary = pd.DataFrame([
        {
            "Analysis": name,
            "Formula": r["formula"],
            **{key: r[key] for key in ["n", "df_effect", "df_resid", "F", "p", "partial_eta_squared", "cohen_f_squared"]},
        }
        for name, r in global_results.items()
    ])
    global_summary.to_csv(table_dir / "global_GMV_WMV_TIV_ANCOVA_summary.csv", index=False)
    return global_results, global_specs, global_summary


@app.cell
def _(bu, global_results, global_summary, mo):
    mo.ui.tabs({
        "Summary": bu.show_table(mo, global_summary),
        **{name: bu.show_table(mo, r["table"].rename_axis("term")) for name, r in global_results.items()},
    })
    return


@app.cell
def _(bu, global_results, global_specs, graph_dir, mo, plt):
    _file_names = {
        "GMV_all": "GMV_birthgroup_differences.png",
        "WMV_all": "WMV_birthgroup_differences.png",
        "TIV_all": "TIV_birthgroup_differences.png",
        "GMV_males": "GMV_birthgroup_differences_males.png",
        "GMV_females": "GMV_birthgroup_differences_females.png",
        "WMV_males": "WMV_birthgroup_differences_males.png",
        "WMV_females": "WMV_birthgroup_differences_females.png",
    }
    _plot_options = {
        "WMV_all": {"yticks": list(range(-60, 121, 20))},
        "TIV_all": {"ylim": (1000, 2200), "zero_line": False},
    }

    group_figures = {}

    for _name, (_data, _formula, _col, _label) in global_specs.items():
        _fig, _ax = plt.subplots(figsize=(70 * bu.MM_TO_INCH, 85 * bu.MM_TO_INCH))
        bu.birthgroup_panel(
            _ax, _data, _col, _label, global_results[_name],
            **{"zero_line": True, **_plot_options.get(_name, {})},
        )
        _fig.tight_layout()
        group_figures[_name] = bu.save_figure(_fig, graph_dir / _file_names[_name])
        plt.close(_fig)

    mo.ui.tabs(group_figures)
    return


@app.cell
def _(bu, df, mo, pd, table_dir):
    _high_ses_females = df[(df["sex"] == 2) & (df["ses0y2"] == 1)]

    _ttest = bu.two_sample_ttest(
        _high_ses_females.loc[_high_ses_females["blsgroup"] == 1, "GMV_resid_har"],
        _high_ses_females.loc[_high_ses_females["blsgroup"] == 2, "GMV_resid_har"],
        "High-SES VP/VLBW females vs high-SES FT females", "GMV_resid_har", "VP/VLBW", "FT",
    )
    pd.DataFrame([_ttest]).to_csv(table_dir / "high_SES_female_GMV_resid_ttest.csv", index=False)

    high_ses_female_summary = _high_ses_females.groupby("blsgroup")["GMV_resid_har"].describe()
    high_ses_female_summary.to_csv(table_dir / "high_SES_female_GMV_resid_by_birthgroup.csv")

    mo.vstack([
        mo.md(
            "### High-SES females: VP/VLBW vs FT (TIV-normalized GMV)\n"
            f"Student t({_ttest['df']:.0f}) = **{_ttest['t']:.4f}**, p = **{_ttest['p']:.4f}** (manuscript) · "
            f"Welch t({_ttest['df_welch']:.1f}) = {_ttest['t_welch']:.4f}, p = {_ttest['p_welch']:.4f}"
        ),
        bu.show_table(mo, high_ses_female_summary),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## DK lobe GMV (Supplementary)

    lobe GMV ~ C(blsgroup) + sex + age + harmonized TIV, Benjamini–Hochberg FDR across the 4 lobes.
    All lobes are carried forward to moderation regardless of significance.
    """)
    return


@app.cell
def _(bu, df, multipletests, pd, table_dir):
    lobe_results = pd.DataFrame([
        {
            "Lobe": bu.lobe_label(col),
            "Variable": col,
            **{key: r[key] for key in ["n", "df_effect", "df_resid", "F", "p", "partial_eta_squared", "cohen_f_squared"]},
        }
        for col in bu.DK_LOBE_GMV_COLS
        for r in [bu.run_ancova(
            df[[col, "blsgroup", "sex", "Age_at_scan2", "TIV_harmonized"]].dropna(),
            f"{col} ~ C(blsgroup) + sex + Age_at_scan2 + TIV_harmonized",
        )]
    ])

    lobe_results["p_fdr"] = multipletests(lobe_results["p"], method="fdr_bh")[1]
    lobe_results["sig_uncorrected"] = lobe_results["p"] < 0.05
    lobe_results["sig_fdr"] = lobe_results["p_fdr"] < 0.05
    lobe_results = lobe_results.sort_values("p")

    lobe_results.to_csv(table_dir / "DK_lobe_GMV_birthgroup_ANCOVA_results.csv", index=False)
    lobe_results.to_csv(table_dir / "DK_lobes_all_for_moderation.csv", index=False)
    (table_dir / "DK_lobes_all_for_moderation.txt").write_text("".join(f"{c}\n" for c in bu.DK_LOBE_GMV_COLS))

    for _kind, _flag in [("FDR", "sig_fdr"), ("uncorrected", "sig_uncorrected")]:
        _sig = lobe_results[lobe_results[_flag]]
        _sig.to_csv(table_dir / f"DK_lobes_significant_birthgroup_{_kind}.csv", index=False)
        (table_dir / f"DK_lobes_for_moderation_{_kind}.txt").write_text("".join(f"{c}\n" for c in _sig["Variable"]))
    return (lobe_results,)


@app.cell
def _(bu, graph_dir, lobe_results, mo, plt, sns):
    fig_lobes, _ax = plt.subplots(figsize=(100 * bu.MM_TO_INCH, 80 * bu.MM_TO_INCH))

    sns.barplot(
        data=lobe_results.sort_values("F", ascending=False), y="Lobe", x="F", hue="sig_fdr",
        dodge=False, palette={True: "black", False: "0.70"}, ax=_ax,
    )
    _ax.set_xlabel("ANCOVA F-statistic for birth group", fontsize=12)
    _ax.set_ylabel("")
    _ax.legend(title="FDR p < .05", loc="lower right", fontsize=12, title_fontsize=12)
    fig_lobes.tight_layout()

    bu.save_figure(fig_lobes, graph_dir / "DK_lobe_GMV_birthgroup_effects.png", dpi=600)

    mo.vstack([
        mo.md(
            f"Significant lobes: **{lobe_results['sig_uncorrected'].sum()}** uncorrected, "
            f"**{lobe_results['sig_fdr'].sum()}** FDR-corrected"
        ),
        bu.show_table(mo, lobe_results),
        fig_lobes,
    ])
    return


if __name__ == "__main__":
    app.run()
