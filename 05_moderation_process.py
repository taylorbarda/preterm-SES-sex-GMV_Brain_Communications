import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 05 · Moderation analyses (PROCESS v4.3 for R)

    | Model | Data | Predictor X | Moderators | Covariates |
    |---|---|---|---|---|
    | **3** (moderated moderation) | whole group | birth group *or* GA (`tz`) | W = SES, Z = sex | age (z) |
    | **1** (simple moderation) | males / females | birth group *or* GA (`tz`) | W = SES | age (z) |

    Outcomes: TIV-normalized GMV, TIV-normalized WMV, harmonized TIV, and the four DK lobe GMVs
    (lobe models also adjust for harmonized TIV, z-scored).
    Every PROCESS text output is saved to `analysis/moderation/` with the same file names as before.

    | | |
    |---|---|
    | **Input** | `NeuroCombat_resid_global_and_lobes_correct.csv` (notebook 04) |
    | **Outputs** | 42 PROCESS `.txt` files → read by notebook 06 |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    return bu, config, mo, pd, plt, sns


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(bu, config, pd):
    df_process = pd.read_csv(config.RESID_FILE)
    bu.require_columns(df_process, bu.DK_LOBE_GMV_COLS, "in moderation data")

    for _col in ["blsgroup", "sex", "ses0y2"]:
        df_process[_col] = pd.to_numeric(df_process[_col], errors="coerce")

    # z-scored covariates for numerical stability
    df_process["TIV_ml"] = df_process["TIV_harmonized"] / 1000
    df_process["Age_at_scan2_z"] = bu.zscore(df_process["Age_at_scan2"])
    df_process["TIV_ml_z"] = bu.zscore(df_process["TIV_ml"])

    (config.MOD_DIR / "tables" / "DK_lobes_all_for_moderation.txt").write_text(
        "".join(f"{col}\n" for col in bu.DK_LOBE_GMV_COLS)
    )

    df_m_process = df_process[df_process["sex"] == 1].copy()
    df_f_process = df_process[df_process["sex"] == 2].copy()
    return df_f_process, df_m_process, df_process


@app.cell
def _(bu, config, df_process, mo):
    _cell_counts = df_process.groupby(["blsgroup", "sex", "ses0y2"]).size().reset_index(name="n")

    _normality = bu.shapiro_table(
        df_process, ["GMV_resid_har", "WMV_resid_har", "TIV_harmonized"] + bu.DK_LOBE_GMV_COLS
    )
    _normality.to_csv(config.MOD_DIR / "tables" / "moderation_outcome_normality.csv", index=False)

    mo.ui.tabs({
        "Cell counts (group × sex × SES)": bu.show_table(mo, _cell_counts),
        "Outcome normality": bu.show_table(mo, _normality),
    })
    return


@app.cell
def _(bu):
    process_jobs = []

    _age = ["Age_at_scan2_z"]
    _age_tiv = ["Age_at_scan2_z", "TIV_ml_z"]

    for _x in ["blsgroup", "tz"]:
        # Global outcomes
        for _y, _short in [("GMV_resid_har", "GMV"), ("WMV_resid_har", "WMV"), ("TIV_harmonized", "TIV")]:
            process_jobs.append(bu.process_job(
                "all", _y, _x, f"{_short}_model3_{_x}_SES_sex_age.txt", model=3, cov=_age, z="sex",
            ))
            for _sex in ["males", "females"]:
                process_jobs.append(bu.process_job(
                    _sex, _y, _x, f"{_y}_model1_{_sex}_{_x}_SES_age.txt", model=1, cov=_age,
                ))

        # DK lobe outcomes (complete cases only, TIV as extra covariate)
        for _lobe in bu.DK_LOBE_GMV_COLS:
            process_jobs.append(bu.process_job(
                "all", _lobe, _x, f"{_lobe}_model3_{_x}_SES_sex_age_TIV.txt",
                model=3, cov=_age_tiv, z="sex", complete_cases=True,
            ))
            for _sex in ["males", "females"]:
                process_jobs.append(bu.process_job(
                    _sex, _lobe, _x, f"{_lobe}_model1_{_sex}_{_x}_SES_age_TIV.txt",
                    model=1, cov=_age_tiv, complete_cases=True,
                ))
    return (process_jobs,)


@app.cell
def _(config, mo, process_jobs):
    mo.md(
        f"**{len(process_jobs)} PROCESS models** · bootstrap samples: {config.PROCESS_BOOT} · seed: {config.PROCESS_SEED}  \n"
        f"PROCESS file: `{config.PROCESS_R_FILE.name}`"
    )
    return


@app.cell
def _(bu, config, df_f_process, df_m_process, df_process, mo, process_jobs):
    with mo.status.spinner(title="Running PROCESS in R…"):
        process_log = bu.run_process_jobs(
            process_jobs,
            datasets={"all": df_process, "males": df_m_process, "females": df_f_process},
            out_dir=config.MOD_DIR,
        )

    _failed = process_log[process_log["status"] != "completed"]

    mo.vstack([
        mo.callout(
            mo.md(f"**{len(process_log) - len(_failed)} / {len(process_log)}** PROCESS models completed"),
            kind="success" if _failed.empty else "danger",
        ),
        bu.show_table(mo, process_log, "PROCESS run log"),
    ])
    return (process_log,)


@app.cell
def _(config, mo, process_log):
    _outputs = sorted(process_log["output_name"])
    output_picker = mo.ui.dropdown(options=_outputs, value=_outputs[0], label="PROCESS output file")
    output_picker
    return (output_picker,)


@app.cell
def _(config, mo, output_picker):
    mo.plain_text((config.MOD_DIR / output_picker.value).read_text(errors="ignore"))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Exploratory interaction plots (raw data)

    Mean ± SE by birth group, SES and sex, with individual participants. These are quick
    checks; the publication figures are built from the PROCESS outputs in notebook 06.
    """)
    return


@app.cell
def _(bu, config, df_process, plt, sns):
    _ses_palette = {1: "#66CDAA", 2: "#B0B0B0", 3: "#CC79A7"}
    _ses_labels = {1: "High SES", 2: "Medium SES", 3: "Low SES"}

    plot_df = df_process.copy()
    plot_df["birth_group"] = plot_df["blsgroup"].map(bu.BIRTHGROUP_LABELS)
    plot_df["sex_label"] = plot_df["sex"].map({1: "Male", 2: "Female"})

    def plot_model3_interaction(y_col, y_label, output_file):
        plot_data = plot_df[["birth_group", "sex_label", "ses0y2", y_col]].dropna()

        fig, axes = plt.subplots(1, 2, figsize=(bu.JOURNAL_WIDTH, 80 * bu.MM_TO_INCH), sharey=True)

        for ax, sex_name in zip(axes, ["Male", "Female"]):
            sex_data = plot_data[plot_data["sex_label"] == sex_name]

            sns.pointplot(
                data=sex_data, x="birth_group", y=y_col, hue="ses0y2",
                order=bu.BIRTHGROUP_ORDER, hue_order=[1, 2, 3], palette=_ses_palette,
                dodge=0.25, errorbar="se", markers="o", capsize=0.12,
                err_kws={"linewidth": 1}, ax=ax,
            )
            sns.stripplot(
                data=sex_data, x="birth_group", y=y_col, order=bu.BIRTHGROUP_ORDER,
                color="black", alpha=0.20, jitter=0.18, size=2, ax=ax,
            )

            ax.set_title(sex_name)
            ax.set_xlabel("Birth group")
            ax.set_ylabel(y_label if ax is axes[0] else "")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend_.remove()

        fig.legend(
            handles[:3], [_ses_labels[int(float(l))] for l in labels[:3]], title="SES",
            loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.13),
        )
        fig.tight_layout()
        bu.save_figure(fig, output_file)
        plt.close(fig)
        return fig

    _graph_dir = config.MOD_DIR / "graphs"
    _specs = [
        ("GMV_resid_ml", "TIV-normalized GMV (mL)", "GMV_model3_birthgroup_SES_sex_interaction.png"),
        ("WMV_resid_ml", "TIV-normalized WMV (mL)", "WMV_model3_birthgroup_SES_sex_interaction.png"),
        ("TIV_ml", "Harmonized TIV (mL)", "TIV_model3_birthgroup_SES_sex_interaction.png"),
    ] + [
        (f"{lobe}_ml", f"{bu.lobe_label(lobe)} lobe GMV (mL)", f"{lobe}_model3_birthgroup_SES_sex_interaction.png")
        for lobe in bu.DK_LOBE_GMV_COLS
    ]

    interaction_figures = {
        label.replace(" (mL)", ""): plot_model3_interaction(col, label, _graph_dir / filename)
        for col, label, filename in _specs
    }
    return (interaction_figures,)


@app.cell
def _(interaction_figures, mo):
    mo.ui.tabs(interaction_figures)
    return


if __name__ == "__main__":
    app.run()
