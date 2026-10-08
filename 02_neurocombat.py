import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 02 · NeuroCombat harmonization

    1. Harmonize global volumes (eTIV, WMV, GMV) across the 4 scanners.
    2. Harmonize the 68 DK cortical + 15 subcortical GMV ROIs, with harmonized TIV as a covariate.
    3. Sum harmonized DK regions into **frontal / parietal / temporal / occipital** lobe GMV.
    4. Check scanner effects with ANCOVAs **before vs after** harmonization (Supplementary Fig. 1).

    | | |
    |---|---|
    | **Input** | `meta_aseg_aparc_subjects_excluded_correct.csv` (notebook 01) |
    | **Output** | `NCH_cortical_subcortical_GMV_ROIs_correct.csv` → notebooks 03 and 04 |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import pingouin as pg
    import seaborn as sns
    from neuroCombat import neuroCombat

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    return bu, config, mo, neuroCombat, pd, pg, plt, sns


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(bu, config, mo, pd):
    df = pd.read_csv(config.EXCLUDED_FILE)

    gmv_rois = [roi for roi in bu.CORTICAL_ROIS + bu.SUBCORTICAL_ROIS if roi in df.columns]
    missing_rois = [roi for roi in bu.CORTICAL_ROIS + bu.SUBCORTICAL_ROIS if roi not in df.columns]

    (config.OUT_DIR / "gmv_roi_names.txt").write_text("".join(f"{roi}\n" for roi in gmv_rois))

    mo.md(
        f"Participants: **{len(df)}** · GMV ROIs used: **{len(gmv_rois)}** "
        f"({sum(r in gmv_rois for r in bu.CORTICAL_ROIS)} cortical, "
        f"{sum(r in gmv_rois for r in bu.SUBCORTICAL_ROIS)} subcortical)"
        + (f"  \n⚠️ Missing ROI columns: {missing_rois}" if missing_rois else "")
    )
    return df, gmv_rois


@app.cell
def _(mo):
    mo.md(r"""
    ## Harmonization

    Batch = `Scanner_ID`; biological covariates preserved: sex, birth group, age
    (+ harmonized TIV for the ROIs).
    """)
    return


@app.cell
def _(bu, df, gmv_rois, neuroCombat, pd):
    def run_neurocombat(data, feature_cols, covars, continuous_cols):
        """NeuroCombat expects features x subjects, so the data are transposed in and out."""
        harmonized = neuroCombat(
            dat=data[feature_cols].T,
            covars=covars,
            batch_col="Scanner_ID",
            categorical_cols=["sex", "blsgroup"],
            continuous_cols=list(continuous_cols),
        )["data"]

        output = pd.DataFrame(harmonized.T, columns=feature_cols)
        output["MRT_ID"] = data["MRT_ID"].values
        return output

    global_cols = {
        "eTIV": "TIV_harmonized",
        "CerebralWhiteMatterVol": "WMV_harmonized",
        "TotalGrayVol": "GMV_harmonized",
    }

    global_combat = run_neurocombat(
        df, list(global_cols), covars=df, continuous_cols=["Age_at_scan2"]
    ).rename(columns=global_cols)

    # Covariates for ROI Combat: raw global + raw ROI columns dropped, harmonized globals added
    roi_covars = (
        df.drop(columns=list(global_cols) + gmv_rois, errors="ignore")
        .merge(global_combat, on="MRT_ID", how="left")
    )

    roi_combat = run_neurocombat(
        df, gmv_rois, covars=roi_covars, continuous_cols=["Age_at_scan2", "TIV_harmonized"]
    )

    harmonized = bu.add_lobe_composites(roi_combat.merge(roi_covars, on="MRT_ID", how="left"))
    raw_with_lobes = bu.add_lobe_composites(df)
    return harmonized, raw_with_lobes


@app.cell
def _(bu, config, harmonized, mo):
    harmonized.to_csv(config.COMBAT_FILE, index=False)

    mo.vstack([
        mo.md(f"Saved harmonized ROI + lobe dataset ({harmonized.shape[0]} × {harmonized.shape[1]}) to "
              f"`{config.COMBAT_FILE.relative_to(config.PROJECT_DIR)}`"),
        bu.show_table(mo, harmonized[["MRT_ID", "TIV_harmonized", "GMV_harmonized", "WMV_harmonized"] + bu.DK_LOBE_GMV_COLS].head(),
                      "Harmonized global and lobe volumes (first rows)"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Scanner effects before vs after NeuroCombat

    ANCOVA (pingouin), scanner as between-subject factor.
    **Before:** volume ~ scanner + sex + age + eTIV · **After:** harmonized volume ~ scanner + sex + age + harmonized TIV.
    """)
    return


@app.cell
def _(pd, pg):
    def scanner_ancova(data, dv, covars):
        test_data = data.dropna(subset=[dv, "Scanner_ID"] + covars).copy()
        test_data[dv] = pd.to_numeric(test_data[dv], errors="coerce")
        test_data = test_data.dropna(subset=[dv])

        ancova = pg.ancova(data=test_data, dv=dv, between="Scanner_ID", covar=covars)
        row = ancova.loc[ancova["Source"] == "Scanner_ID"].iloc[0]
        p_col = next(col for col in ["p-unc", "p_unc", "pval", "p-value", "p"] if col in ancova.columns)

        return {"F": row["F"], "p": row[p_col], "n": len(test_data)}

    def compare_scanner_effects(raw_data, harmonized_data, items, record_variable=True):
        """
        items: list of (labels, before_col, after_col).
        Returns one row per item and stage (Before/After).
        """
        rows = []

        for labels, before_col, after_col in items:
            for stage, data, col, covars in [
                ("Before", raw_data, before_col, ["sex", "Age_at_scan2", "eTIV"]),
                ("After", harmonized_data, after_col, ["sex", "Age_at_scan2", "TIV_harmonized"]),
            ]:
                row = dict(labels)
                if record_variable:
                    row["Variable"] = col
                row["Stage"] = stage
                row.update(scanner_ancova(data, col, covars))
                rows.append(row)

        return pd.DataFrame(rows)

    def summarize_scanner_effects(results, index_cols):
        summary = results.pivot(index=index_cols, columns="Stage", values=["F", "p", "n"]).reset_index()
        summary.columns = ["_".join(col).strip("_") for col in summary.columns]

        summary["scanner_sig_before"] = summary["p_Before"] < 0.05
        summary["scanner_sig_after"] = summary["p_After"] < 0.05
        summary["scanner_effect_removed"] = summary["scanner_sig_before"] & ~summary["scanner_sig_after"]

        return summary
    return compare_scanner_effects, summarize_scanner_effects


@app.cell
def _(bu, compare_scanner_effects, config, harmonized, raw_with_lobes, summarize_scanner_effects):
    lobe_scanner_results = compare_scanner_effects(
        raw_with_lobes,
        harmonized,
        [({"Lobe": bu.lobe_label(col)}, col, col) for col in bu.DK_LOBE_GMV_COLS],
    )
    lobe_scanner_summary = summarize_scanner_effects(lobe_scanner_results, ["Lobe", "Variable"])

    lobe_scanner_results.to_csv(config.OUT_DIR / "DK_lobe_scanner_effects_before_after_NeuroCombat.csv", index=False)
    lobe_scanner_summary.to_csv(config.OUT_DIR / "DK_lobe_scanner_effects_summary_before_after_NeuroCombat.csv", index=False)
    return lobe_scanner_results, lobe_scanner_summary


@app.cell
def _(bu, config, lobe_scanner_results, lobe_scanner_summary, mo):
    fig_lobe_scanner = bu.scanner_effect_barplot(
        lobe_scanner_results,
        label_col="Lobe",
        title="DK Lobe GMV: Scanner Effects Before and After NeuroCombat",
        figsize=(110 * bu.MM_TO_INCH, 85 * bu.MM_TO_INCH),
        output_file=config.OUT_DIR / "DK_lobe_scanner_effects_before_after_NeuroCombat.png",
        dpi=600,
    )

    mo.vstack([
        mo.md("### DK lobes"),
        bu.show_table(mo, lobe_scanner_summary),
        fig_lobe_scanner,
    ])
    return


@app.cell
def _(compare_scanner_effects, config, df, harmonized):
    global_scanner_results = compare_scanner_effects(
        df,
        harmonized,
        [
            ({"Volume Type": "GMV"}, "TotalGrayVol", "GMV_harmonized"),
            ({"Volume Type": "WMV"}, "CerebralWhiteMatterVol", "WMV_harmonized"),
        ],
    )
    global_scanner_results.to_csv(
        config.OUT_DIR / "global_GMV_WMV_scanner_effects_before_after_NeuroCombat.csv", index=False
    )
    return (global_scanner_results,)


@app.cell
def _(bu, config, global_scanner_results, mo, plt, sns):
    fig_global_scanner, _ax = plt.subplots(figsize=(8, 5))

    sns.barplot(
        data=global_scanner_results, x="Volume Type", y="F", hue="Stage",
        hue_order=["Before", "After"], palette=bu.SCANNER_STAGE_PALETTE, ax=_ax,
    )
    for _container in _ax.containers:
        _ax.bar_label(_container, fmt="%.2f", fontsize=12, padding=3)

    _ax.set_xlabel("Volume Type", fontsize=12)
    _ax.set_ylabel("F-statistic (Scanner_ID)", fontsize=12)
    _ax.tick_params(axis="both", labelsize=12)
    _ax.legend(title="Stage", fontsize=12, title_fontsize=12, frameon=True)
    sns.despine(ax=_ax)
    fig_global_scanner.tight_layout()

    bu.save_figure(
        fig_global_scanner,
        config.OUT_DIR / "global_GMV_WMV_scanner_effects_before_after_NeuroCombat.png",
        dpi=600,
    )

    mo.vstack([
        mo.md("### Global GMV / WMV (Supplementary Fig. 1)"),
        bu.show_table(mo, global_scanner_results),
        fig_global_scanner,
    ])
    return


@app.cell
def _(compare_scanner_effects, config, df, gmv_rois, harmonized, summarize_scanner_effects):
    roi_scanner_results = compare_scanner_effects(
        df,
        harmonized,
        [
            ({"ROI": roi, "ROI_type": "Cortical" if roi.startswith(("lh_", "rh_")) else "Subcortical"}, roi, roi)
            for roi in gmv_rois
        ],
        record_variable=False,
    )
    roi_scanner_summary = summarize_scanner_effects(roi_scanner_results, ["ROI", "ROI_type"])

    roi_scanner_results.to_csv(config.OUT_DIR / "ROI_scanner_effects_before_after_NeuroCombat.csv", index=False)
    roi_scanner_summary.to_csv(config.OUT_DIR / "ROI_scanner_effects_summary_before_after_NeuroCombat.csv", index=False)
    return roi_scanner_results, roi_scanner_summary


@app.cell
def _(bu, config, mo, roi_scanner_results, roi_scanner_summary):
    roi_figures = {
        roi_type: bu.scanner_effect_barplot(
            roi_scanner_results[roi_scanner_results["ROI_type"] == roi_type],
            label_col="ROI",
            title=f"{roi_type} GMV ROIs: Scanner Effects Before and After NeuroCombat",
            max_bars=max_bars,
            output_file=config.OUT_DIR / filename,
        )
        for roi_type, max_bars, filename in [
            ("Cortical", 30, "cortical_GMV_ROI_scanner_effects_top30.png"),
            ("Subcortical", None, "subcortical_GMV_ROI_scanner_effects.png"),
        ]
    }

    mo.vstack([
        mo.md(
            f"### Individual ROIs\n"
            f"Scanner significant before: **{roi_scanner_summary['scanner_sig_before'].sum()}** / {len(roi_scanner_summary)} · "
            f"after: **{roi_scanner_summary['scanner_sig_after'].sum()}** · "
            f"removed: **{roi_scanner_summary['scanner_effect_removed'].sum()}**"
        ),
        mo.ui.tabs({
            "Summary table": bu.show_table(mo, roi_scanner_summary),
            "Cortical (top 30)": roi_figures["Cortical"],
            "Subcortical": roi_figures["Subcortical"],
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
