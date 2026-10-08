import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 06 · Publication figures and FSIQ models

    | Output | Manuscript |
    |---|---|
    | `Figure1_GMV_WMV_TIV_birthgroup` | Fig. 1 |
    | `Figure_moderated_moderation_GMV_birthgroup_SES_sex` | Fig. 2 |
    | `Figure3_GMV_FSIQ_VPVLBW_partial_regression` | Fig. 3 |
    | `PROCESS_GMV_gestational_age_SES_interactions` | Supplementary Fig. 2 |
    | `PROCESS_WMV_TIV_*_SES_interactions` | Supplementary Fig. 3 |
    | `fsiq_gmv_ols_*.txt` | FSIQ regressions, Supplementary Table 6 |
    | `supplementary_lobes/*` | DK lobe PROCESS figures and Int_4 tables |

    | | |
    |---|---|
    | **Inputs** | `NeuroCombat_resid_global_and_lobes_correct.csv` (notebook 04), PROCESS outputs (notebook 05) |
    | **Outputs** | `analysis/graphs_publication/` |
    """)
    return


@app.cell
def _():
    import math

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from scipy.stats import pearsonr

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

    MM = bu.MM_TO_INCH
    pub_dir = config.PUB_DIR
    supp_dir = config.SUPP_DIR
    process_dir = config.MOD_DIR
    return MM, bu, config, math, mo, np, pd, pearsonr, plt, process_dir, pub_dir, sm, smf, sns, supp_dir


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(bu, config, pd):
    data = pd.read_csv(config.RESID_FILE)
    bu.require_columns(data, bu.DK_LOBE_GMV_COLS, "in visualization data")

    data["birth_group"] = data["blsgroup"].map(bu.BIRTHGROUP_LABELS)
    data["GMV_resid_har_ml"] = data["GMV_resid_har"] / 1000
    data["WMV_resid_har_ml"] = data["WMV_resid_har"] / 1000
    data["TIV_100ml"] = data["TIV_harmonized"] / 100000
    return (data,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 1 · GMV, WMV and TIV by birth group
    ANCOVA: outcome ~ C(blsgroup) + sex + age
    """)
    return


@app.cell
def _(MM, bu, data, plt, pub_dir):
    _panels = [
        ("GMV_resid_har", "GMV_resid_har_ml", "TIV-normalized GMV (mL)", "A", None),
        ("WMV_resid_har", "WMV_resid_har_ml", "TIV-normalized WMV (mL)", "B", None),
        ("TIV_harmonized", "TIV_100ml", "TIV (×100 mL)", "C", (10, 23)),
    ]

    fig1, _axes = plt.subplots(1, 3, figsize=(bu.JOURNAL_WIDTH * 1.5, 85 * MM))

    for _ax, (_outcome, _col, _label, _panel, _ylim) in zip(_axes, _panels):
        bu.birthgroup_panel(
            _ax, data, _col, _label,
            bu.run_ancova(data, f"{_outcome} ~ C(blsgroup) + sex + Age_at_scan2"),
            panel_label=_panel, ylim=_ylim, zero_line="normalized" in _label,
            box_linewidth=0.8, point_size=2.2, stat_fontsize=10, stat_height=0.10, stat_sep=",",
        )
        _ax.title.set_fontsize(12)
        _ax.set_ylabel(_label, fontsize=12, labelpad=8)
        _ax.set_xlabel("Birth Group", fontsize=12, labelpad=8)
        _ax.tick_params(axis="y", labelsize=10)
        _ax.tick_params(axis="x", labelsize=10, pad=6)
        _ax.grid(True, axis="y", linewidth=0.6, alpha=0.55)
        _ax.grid(True, axis="x", linewidth=0.4, alpha=0.30)

    fig1.tight_layout(w_pad=4.0)
    bu.save_figure(fig1, pub_dir / "Figure1_GMV_WMV_TIV_birthgroup.png", dpi=600, tiff=True)
    fig1
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PROCESS conditional-effect figures
    Lines are PROCESS model-implied values at each SES level (from the "Data for visualizing" block).
    """)
    return


@app.cell
def _(bu, process_dir):
    def load_process_set(outcome, x, model1_prefix):
        """Whole group (model 3) + females/males (model 1) plot tables for one outcome and predictor."""
        return {
            "Whole Group": bu.parse_process_plot_data(process_dir / f"{outcome}_model3_{x}_SES_sex_age.txt"),
            "Females": bu.parse_process_plot_data(process_dir / f"{model1_prefix}_model1_females_{x}_SES_age.txt"),
            "Males": bu.parse_process_plot_data(process_dir / f"{model1_prefix}_model1_males_{x}_SES_age.txt"),
        }

    process_plot_data = {
        (outcome, x): load_process_set(outcome, x, prefix)
        for outcome, prefix in [("GMV", "GMV_resid_har"), ("WMV", "WMV_resid_har"), ("TIV", "TIV_harmonized")]
        for x in ["blsgroup", "tz"]
    }
    return (process_plot_data,)


@app.cell
def _(bu, np, plt, process_plot_data):
    OUTCOME_SPECS = {
        # outcome: (PROCESS y column, y scale, y label)
        "GMV": ("GMV_resid_har", 1000, "TIV-normalized GMV (mL)"),
        "WMV": ("WMV_resid_har", 1000, "TIV-normalized WMV (mL)"),
        "TIV": ("TIV_harmonized", 100000, "TIV (×100 mL)"),
    }

    def process_panel_figure(
        outcomes,
        x_col,
        figsize,
        sharey=False,
        ylim=None,
        y_labels=None,
        restyle=None,
        legend=None,
        layout=None,
    ):
        """
        One row per outcome, columns = Whole Group / Females / Males, panels labelled A, B, C, ...
        """
        fig, axes = plt.subplots(len(outcomes), 3, figsize=figsize, sharey=sharey, squeeze=False)
        panel_letters = iter("ABCDEFGHI")
        y_labels = y_labels or {}

        for row, outcome in enumerate(outcomes):
            y_col, y_scale, y_label = OUTCOME_SPECS[outcome]
            y_label = y_labels.get(outcome, y_label)

            for col, group in enumerate(["Whole Group", "Females", "Males"]):
                bu.plot_process_ses(
                    axes[row, col], process_plot_data[(outcome, x_col)][group], x_col=x_col,
                    y_col=y_col, title=group, y_label=y_label if col == 0 else "",
                    panel_label=next(panel_letters), y_scale=y_scale, ylim=ylim,
                )

        bu.restyle_process_axes(axes, **(restyle or {}))

        if (restyle or {}).get("publication", True):
            for row, outcome in enumerate(outcomes):
                axes[row, 0].set_ylabel(y_labels.get(outcome, OUTCOME_SPECS[outcome][2]), fontsize=12, labelpad=10)

            for ax in np.ravel(axes):
                ax.xaxis.labelpad = 6

        bu.ses_legend(fig, **(legend or {}))
        fig.tight_layout(**(layout or {}))
        return fig
    return (process_panel_figure,)


@app.cell
def _(bu, process_panel_figure, pub_dir):
    fig_gmv_bg = process_panel_figure(
        ["GMV"], "blsgroup", figsize=(bu.JOURNAL_WIDTH * 1.25, 95 * bu.MM_TO_INCH), sharey=True,
        ylim=(-16, 10), legend={"bbox_to_anchor": (0.5, 1.08)},
        layout={"w_pad": 2.0, "rect": [0.02, 0, 1, 0.88]},
    )
    bu.save_figure(fig_gmv_bg, pub_dir / "PROCESS_GMV_birthgroup_SES_interactions.png", dpi=600)

    fig_wmv_tiv_bg = process_panel_figure(
        ["WMV", "TIV"], "blsgroup", figsize=(bu.JOURNAL_WIDTH * 1.25, 150 * bu.MM_TO_INCH),
        restyle={"panel_pos": (-0.25, 1.1)}, legend={"bbox_to_anchor": (0.5, 1.04)},
        layout={"w_pad": 2.0, "h_pad": 2.2, "rect": [0.02, 0, 1, 0.92]},
    )
    bu.save_figure(fig_wmv_tiv_bg, pub_dir / "PROCESS_WMV_TIV_birthgroup_SES_interactions.png", dpi=600)

    fig_gmv_tz = process_panel_figure(
        ["GMV"], "tz", figsize=(bu.JOURNAL_WIDTH * 1.25, 95 * bu.MM_TO_INCH), sharey=True,
        ylim=(-20, 12), legend={"bbox_to_anchor": (0.5, 1.08)},
        layout={"w_pad": 2.0, "rect": [0.02, 0, 1, 0.88]},
    )
    bu.save_figure(fig_gmv_tz, pub_dir / "PROCESS_GMV_gestational_age_SES_interactions.png", dpi=600, tiff=True)

    fig_wmv_tiv_tz = process_panel_figure(
        ["WMV", "TIV"], "tz", figsize=(190 * bu.MM_TO_INCH, 135 * bu.MM_TO_INCH),
        y_labels={"TIV": "TIV (mL) ×10²"},
        restyle={"tick_size": 10, "label_size": 10, "panel_size": 12, "panel_pos": (-0.15, 1.3), "publication": False},
        legend={"bbox_to_anchor": (0.5, 1.01), "linewidth": 1.1, "markersize": 3.5},
        layout={"rect": [0.03, 0, 1, 0.955], "w_pad": 1.2, "h_pad": 1.6},
    )
    bu.save_figure(fig_wmv_tiv_tz, pub_dir / "PROCESS_WMV_TIV_gestational_age_SES_interactions_1legend.png", dpi=600, tiff=True)
    return fig_gmv_bg, fig_gmv_tz, fig_wmv_tiv_bg, fig_wmv_tiv_tz


@app.cell
def _(fig_gmv_bg, fig_gmv_tz, fig_wmv_tiv_bg, fig_wmv_tiv_tz, mo):
    mo.ui.tabs({
        "GMV × birth group": fig_gmv_bg,
        "WMV/TIV × birth group (Supp. Fig. 3)": fig_wmv_tiv_bg,
        "GMV × GA (Supp. Fig. 2)": fig_gmv_tz,
        "WMV/TIV × GA": fig_wmv_tiv_tz,
    })
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 2 · Moderated moderation
    Conceptual diagram (A) and PROCESS birth group × SES lines for the whole group (B), females (C) and males (D).
    """)
    return


@app.cell
def _(MM, bu, plt, process_plot_data, pub_dir):
    def draw_moderated_moderation_diagram(ax):
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.axis("off")

        boxes = {
            "Sex": (-0.07, 0.51, 0.28, 0.16),
            "Infant SES": (0.29, 0.70, 0.40, 0.20),
            "Premature\nBirth": (-0.06, 0.18, 0.40, 0.20),
            "Adult GMV": (0.66, 0.18, 0.40, 0.20),
        }

        for label, (x, y, w, h) in boxes.items():
            ax.add_patch(plt.Rectangle((x, y), w, h, facecolor="white", edgecolor="black", linewidth=1.4))
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                    fontsize=10, fontweight="bold", linespacing=0.9)

        ax.text(0.50, 0.96, "Moderated Moderation", ha="center", va="center", fontsize=11, fontweight="bold")

        arrow = dict(arrowstyle="-|>", linewidth=1.5, color="black", shrinkA=4, shrinkB=4)
        birth_mid = (0.50, 0.28)
        ses_start = (0.50, 0.70)
        ses_mid = ((ses_start[0] + birth_mid[0]) / 2, (ses_start[1] + birth_mid[1]) / 2)

        ax.annotate("", xy=(0.68, 0.28), xytext=(0.33, 0.28), arrowprops=arrow)  # birth -> GMV
        ax.annotate("", xy=birth_mid, xytext=ses_start, arrowprops=arrow)        # SES -> path
        ax.annotate("", xy=ses_mid, xytext=(0.21, 0.57), arrowprops=arrow)       # sex -> SES arrow

        ax.text(-0.22, 1.04, "A", transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig2 = plt.figure(figsize=(180 * MM, 155 * MM))
    _gs = fig2.add_gridspec(2, 2, wspace=0.32, hspace=0.42)
    _ax_a = fig2.add_subplot(_gs[0, 0])
    _panel_axes = [fig2.add_subplot(_gs[0, 1]), fig2.add_subplot(_gs[1, 0]), fig2.add_subplot(_gs[1, 1])]

    draw_moderated_moderation_diagram(_ax_a)

    for _ax, _group, _letter in zip(_panel_axes, ["Whole Group", "Females", "Males"], "BCD"):
        bu.plot_process_ses(
            _ax, process_plot_data[("GMV", "blsgroup")][_group], x_col="blsgroup",
            y_col="GMV_resid_har", title=_group, y_label="TIV-normalized GMV (mL)",
            panel_label=_letter, y_scale=1000, ylim=(-16, 10),
        )

    bu.restyle_process_axes(
        _panel_axes, tick_size=10, label_size=10, panel_size=12, panel_pos=(-0.22, 1.04), publication=False,
    )
    bu.ses_legend(fig2, bbox_to_anchor=(0.68, 0.995), fontsize=10, linewidth=1.1, markersize=3.5)

    bu.save_figure(fig2, pub_dir / "Figure_moderated_moderation_GMV_birthgroup_SES_sex.png", dpi=600, tiff=True)
    fig2
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figure 3 · GMV and FSIQ
    Partial regression in the VP/VLBW group (age and sex partialled out), plus the OLS models
    reported in the Results.
    """)
    return


@app.cell
def _(MM, bu, data, pd, pearsonr, plt, pub_dir, sm, sns):
    _preterm = data.loc[data["blsgroup"] == 1, ["GMV_resid_har", "wie_gesiq", "Age_at_scan2", "sex"]].dropna()
    _X = sm.add_constant(_preterm[["Age_at_scan2", "sex"]])

    _resid = pd.DataFrame({
        "GMV_resid_ml": sm.OLS(_preterm["GMV_resid_har"], _X).fit().resid / 1000,
        "FSIQ_resid": sm.OLS(_preterm["wie_gesiq"], _X).fit().resid,
    })

    partial_r, _p_two = pearsonr(_resid["GMV_resid_ml"], _resid["FSIQ_resid"])
    partial_p_one = _p_two / 2 if partial_r > 0 else 1 - _p_two / 2

    fig3, _ax = plt.subplots(figsize=(95 * MM, 80 * MM))
    sns.regplot(
        data=_resid, x="GMV_resid_ml", y="FSIQ_resid", ci=95,
        scatter_kws={"alpha": 0.70, "s": 18, "color": "#1f77b4"},
        line_kws={"color": "red", "linewidth": 1.2}, ax=_ax,
    )
    _ax.text(
        0.05, 0.95, f"Pearson's r = {partial_r:.2f}\n{bu.format_p(partial_p_one)}",
        transform=_ax.transAxes, fontsize=10, va="top",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )
    _ax.set_title("VP/VLBW Group", fontsize=12)
    _ax.set_xlabel("TIV-normalized GMV (mL)", fontsize=12)
    _ax.set_ylabel("FSIQ (a.u.)", fontsize=12)
    _ax.tick_params(axis="both", labelsize=10)
    _ax.grid(True, axis="y", linewidth=0.6, alpha=0.55)
    _ax.grid(True, axis="x", linewidth=0.4, alpha=0.30)
    fig3.tight_layout()

    bu.save_figure(fig3, pub_dir / "Figure3_GMV_FSIQ_VPVLBW_partial_regression.png", dpi=600, tiff=True)
    partial_n = len(_resid)
    return fig3, partial_n, partial_p_one, partial_r


@app.cell
def _(config, data, pd, smf):
    def one_tailed(model, term):
        """p for the directional hypothesis coefficient > 0"""
        t = model.tvalues[term]
        p_two = model.pvalues[term]
        return p_two / 2 if t > 0 else 1 - p_two / 2

    _fsiq = data[["wie_gesiq", "GMV_resid_har", "Age_at_scan2", "sex", "blsgroup"]].dropna()
    _preterm = _fsiq[_fsiq["blsgroup"] == 1]

    fsiq_models = {
        "VP/VLBW: FSIQ ~ GMV + age + sex (manuscript)": smf.ols("wie_gesiq ~ GMV_resid_har + Age_at_scan2 + sex", _preterm).fit(),
        "VP/VLBW: FSIQ ~ GMV + age (as in fsiq_gmv_ols_preterms.txt)": smf.ols("wie_gesiq ~ GMV_resid_har + Age_at_scan2", _preterm).fit(),
        "Full sample: FSIQ ~ GMV × birth group + age + sex (Supp. Table 6)": smf.ols(
            "wie_gesiq ~ GMV_resid_har * C(blsgroup) + Age_at_scan2 + sex", _fsiq
        ).fit(),
    }

    fsiq_summary = pd.DataFrame([
        {
            "Model": name,
            "Term": term,
            "b": model.params[term],
            "SE": model.bse[term],
            "t": model.tvalues[term],
            "df_resid": model.df_resid,
            "p (two-tailed)": model.pvalues[term],
            "p (one-tailed, b > 0)": one_tailed(model, term) if term == "GMV_resid_har" else None,
            "n": int(model.nobs),
        }
        for name, model in fsiq_models.items()
        for term in model.params.index
        if term != "Intercept"
    ])

    _preterm_text = "\n\n".join(
        f"{name}\n{model.summary()}\n\nOne-tailed p for GMV_resid_har (b > 0): {one_tailed(model, 'GMV_resid_har'):.4f}"
        for name, model in list(fsiq_models.items())[:2]
    )
    (config.ANALYSIS_DIR / "fsiq_gmv_ols_preterms.txt").write_text(_preterm_text)
    (config.ANALYSIS_DIR / "fsiq_gmv_ols_full_sample_interaction.txt").write_text(
        str(list(fsiq_models.values())[2].summary())
    )
    fsiq_summary.to_csv(config.ANALYSIS_DIR / "fsiq_gmv_ols_coefficients.csv", index=False)
    return (fsiq_summary,)


@app.cell
def _(bu, fig3, fsiq_summary, mo, partial_n, partial_p_one, partial_r):
    mo.vstack([
        fig3,
        mo.md(f"Partial Pearson r = **{partial_r:.3f}**, one-tailed p = **{partial_p_one:.4f}**, n = {partial_n}"),
        bu.show_table(mo, fsiq_summary, "FSIQ OLS models"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Supplementary · DK lobe moderation
    PROCESS lines for each lobe (model 3, whole group) and the birth group × SES × sex term (Int_4).
    """)
    return


@app.cell
def _(bu, math, plt, supp_dir):
    def plot_lobe_process_grid(lobe_files, x_col, output_filename):
        """2-column grid of PROCESS lines, one panel per lobe file (max 4)."""
        n_rows = math.ceil(len(lobe_files) / 2)
        fig, axes = plt.subplots(n_rows, 2, figsize=(bu.JOURNAL_WIDTH, 110 * bu.MM_TO_INCH), squeeze=False)

        for ax, file_path, letter in zip(axes.flat, lobe_files, "ABCD"):
            plot_df = bu.parse_process_plot_data(file_path)
            lobe = file_path.name.split("_model")[0]

            bu.plot_process_ses(
                ax, plot_df, x_col=x_col, y_col=bu.process_outcome_col(plot_df),
                title=bu.lobe_label(lobe), y_label="GMV (mL)", y_scale=1000,
            )
            ax.title.set_fontsize(12)
            ax.xaxis.label.set_size(10)
            ax.yaxis.label.set_size(10)
            ax.tick_params(axis="both", labelsize=10)
            ax.text(-0.23, 1.13, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top", ha="left")

        for ax in axes.flat[len(lobe_files):]:
            ax.axis("off")

        bu.ses_legend(fig, bbox_to_anchor=(0.5, 1.04), linewidth=1.2, markersize=4)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        bu.save_figure(fig, supp_dir / output_filename, dpi=600, tiff=True)
        plt.close(fig)
        return fig
    return (plot_lobe_process_grid,)


@app.cell
def _(plot_lobe_process_grid, process_dir):
    lobe_bg_files = sorted(process_dir.glob("*_DK_GMV_model3_blsgroup_SES_sex_age_TIV.txt"))
    lobe_tz_files = sorted(process_dir.glob("*_DK_GMV_model3_tz_SES_sex_age_TIV.txt"))

    lobe_grid_figures = {
        "Birth group × SES × sex": plot_lobe_process_grid(
            lobe_bg_files, "blsgroup", "Supplementary_DK_lobes_model3_birthgroup_SES_sex.png"
        ),
        "GA × SES × sex": plot_lobe_process_grid(
            lobe_tz_files, "tz", "Supplementary_DK_lobes_model3_tz_SES_sex.png"
        ),
    }
    return lobe_bg_files, lobe_grid_figures


@app.cell
def _(bu, lobe_bg_files, np, pd, supp_dir):
    _rows = []

    for _file in lobe_bg_files:
        _lobe = _file.name.split("_model3_")[0]
        _term = bu.extract_process_model_term(_file, "Int_4")

        _rows.append({
            "Lobe": bu.lobe_label(_lobe),
            "Variable": _lobe,
            **(_term or {"term": "Int_4", "coeff": np.nan, "se": np.nan, "t": np.nan, "p": np.nan, "LLCI": np.nan}),
            "sig_p05": bool(_term and _term["p"] < 0.05),
            "source_file": _file.name,
            "status": "ok" if _term else "term_not_found",
        })

    lobe_int4 = pd.DataFrame(_rows).sort_values(["sig_p05", "p"], ascending=[False, True])
    sig_lobe_int4 = lobe_int4[lobe_int4["sig_p05"]]

    lobe_int4.to_csv(supp_dir / "Supplementary_DK_lobes_model3_Int4_blsgroup_SES_sex_interactions.csv", index=False)
    sig_lobe_int4.to_csv(supp_dir / "Supplementary_DK_lobes_model3_SIGNIFICANT_Int4_blsgroup_SES_sex_interactions.csv", index=False)
    return lobe_int4, sig_lobe_int4


@app.cell
def _(MM, bu, plot_lobe_process_grid, plt, process_dir, sig_lobe_int4, supp_dir):
    _sig_files = [process_dir / name for name in sig_lobe_int4["source_file"]]

    if len(_sig_files) == 1:
        _plot_df = bu.parse_process_plot_data(_sig_files[0])
        sig_int4_figure, _ax = plt.subplots(figsize=(85 * MM, 75 * MM))
        bu.plot_process_ses(
            _ax, _plot_df, x_col="blsgroup", y_col=bu.process_outcome_col(_plot_df),
            title=sig_lobe_int4["Lobe"].iloc[0], y_label="GMV (mL)", y_scale=1000,
        )
        _ax.legend(
            handles=[plt.Line2D([0], [0], color=bu.SES_PALETTE[s], marker="o", label=s) for s in bu.SES_ORDER],
            title="SES Level", fontsize=7, title_fontsize=7.5, loc="best", frameon=True,
        )
        sig_int4_figure.tight_layout()
        bu.save_figure(sig_int4_figure, supp_dir / "Supplementary_SIGNIFICANT_Int4_DK_lobe_birthgroup_SES_sex_single.png")
        plt.close(sig_int4_figure)
    elif len(_sig_files) > 1:
        sig_int4_figure = plot_lobe_process_grid(
            _sig_files, "blsgroup", "Supplementary_SIGNIFICANT_Int4_DK_lobes_birthgroup_SES_sex.png"
        )
    else:
        sig_int4_figure = None
    return (sig_int4_figure,)


@app.cell
def _(bu, lobe_grid_figures, lobe_int4, mo, sig_int4_figure):
    mo.vstack([
        bu.show_table(mo, lobe_int4, "Int_4 (birth group × SES × sex) per lobe"),
        mo.ui.tabs({
            **lobe_grid_figures,
            "Significant Int_4": sig_int4_figure if sig_int4_figure is not None else mo.md("No lobe has a significant Int_4 (p < .05)."),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
