import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 03 · Descriptive statistics (Table 1)

    Sample characteristics by birth group, two-sample t-tests (Student's, as reported in the
    manuscript, with Welch's alongside), χ² test for sex, Shapiro-Wilk
    normality checks and the FSIQ group plot.

    | | |
    |---|---|
    | **Input** | `NCH_cortical_subcortical_GMV_ROIs_correct.csv` (notebook 02) |
    | **Outputs** | `analysis/descriptive_stats/*.csv`, `FullScaleIQ_by_BirthGroup.png` |
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from scipy.stats import chi2_contingency

    import bc_utils as bu
    import config

    config.ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    return bu, chi2_contingency, config, mo, pd, plt, sns


@app.cell
def _(bu, mo):
    bu.mode_banner(mo)
    return


@app.cell
def _(bu, config, mo, pd):
    data = pd.read_csv(config.COMBAT_FILE)
    bu.require_columns(data, bu.DK_LOBE_GMV_COLS, "in harmonized data")

    # Table 1 uses the long group label
    group_labels = {1: "VP/VLBW", 2: "Full-term"}
    sex_labels = {1: "Male", 2: "Female"}

    vpt = data["blsgroup"] == 1
    ft = data["blsgroup"] == 2
    male = data["sex"] == 1
    female = data["sex"] == 2

    _counts = pd.crosstab(data["blsgroup"].map(group_labels), data["sex"].map(sex_labels), margins=True)

    mo.vstack([
        mo.md(f"**{len(data)}** participants"),
        bu.show_table(mo, _counts, "Participants by group and sex"),
    ])
    return data, female, ft, group_labels, male, sex_labels, vpt


@app.cell
def _(bu, config, data, ft, group_labels, mo, pd, sex_labels, vpt):
    measures = [
        "Age_at_scan2", "sex", "TIV_harmonized", "inti", "tz", "gew", "ses0y2", "wie_gesiq",
    ] + bu.DK_LOBE_GMV_COLS

    available_measures = [m for m in measures if m in data.columns]
    continuous_measures = [m for m in available_measures if m != "sex"]

    describe_preterm = data.loc[vpt, available_measures].describe()
    describe_fullterm = data.loc[ft, available_measures].describe()

    describe_preterm.to_csv(config.DESC_DIR / "DescriptiveStatsPreterm.csv")
    describe_fullterm.to_csv(config.DESC_DIR / "DescriptiveStatsFullterm.csv")

    descriptive_summary = pd.DataFrame([
        {
            "Variable": var,
            "Group": group_name,
            "n": len(values),
            "Mean": values.mean(),
            "SD": values.std(),
            "Median": values.median(),
            "Min": values.min(),
            "Max": values.max(),
        }
        for var in continuous_measures
        for group_value, group_name in group_labels.items()
        for values in [pd.to_numeric(data.loc[data["blsgroup"] == group_value, var], errors="coerce").dropna()]
    ])

    sex_summary = data.groupby(["blsgroup", "sex"]).size().reset_index(name="n")
    sex_summary["Group"] = sex_summary["blsgroup"].map(group_labels)
    sex_summary["Sex"] = sex_summary["sex"].map(sex_labels)
    sex_summary = sex_summary[["Group", "Sex", "n"]]

    descriptive_summary.to_csv(config.DESC_DIR / "DescriptiveSummary_ByGroup.csv", index=False)
    sex_summary.to_csv(config.DESC_DIR / "SexCounts_ByGroup.csv", index=False)

    mo.ui.tabs({
        "Summary by group": bu.show_table(mo, descriptive_summary),
        "Sex counts": bu.show_table(mo, sex_summary),
        "describe(): VP/VLBW": bu.show_table(mo, describe_preterm.rename_axis("stat")),
        "describe(): Full-term": bu.show_table(mo, describe_fullterm.rename_axis("stat")),
    })
    return (continuous_measures,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Group comparisons

    Student's t-tests (columns `t`, `df`, `p`, as in the manuscript) with Welch's t-tests alongside
    (`t_welch`, `df_welch`, `p_welch`) for every continuous measure (VP/VLBW vs full-term), plus SES comparisons
    within sex and within the VP/VLBW group. Sex distribution: χ² test.
    """)
    return


@app.cell
def _(bu, chi2_contingency, config, continuous_measures, data, female, ft, male, mo, pd, vpt):
    _ses_comparisons = [
        ("VP/VLBW females vs Full-term females", vpt & female, ft & female, "VP/VLBW females", "Full-term females"),
        ("VP/VLBW males vs Full-term males", vpt & male, ft & male, "VP/VLBW males", "Full-term males"),
        ("Males vs females", male, female, "Males", "Females"),
        ("VP/VLBW males vs VP/VLBW females", vpt & male, vpt & female, "VP/VLBW males", "VP/VLBW females"),
    ]

    all_ttests = pd.DataFrame(
        [
            bu.two_sample_ttest(data.loc[vpt, var], data.loc[ft, var], "VP/VLBW vs Full-term", var, "VP/VLBW", "Full-term")
            for var in continuous_measures
        ]
        + [
            bu.two_sample_ttest(data.loc[mask_1, "ses0y2"], data.loc[mask_2, "ses0y2"], label, "ses0y2", g1, g2)
            for label, mask_1, mask_2, g1, g2 in _ses_comparisons
        ]
    )

    sex_contingency = pd.crosstab(data["sex"], data["blsgroup"])
    _chi2, _p, _dof, _expected = chi2_contingency(sex_contingency)

    sex_test = pd.DataFrame([{
        "Comparison": "Sex distribution by birth group",
        "Variable": "sex",
        "Test": "Chi-square test",
        "chi2": _chi2,
        "df": _dof,
        "p": _p,
        "VPVLBW_male_n": (vpt & male).sum(),
        "VPVLBW_female_n": (vpt & female).sum(),
        "Fullterm_male_n": (ft & male).sum(),
        "Fullterm_female_n": (ft & female).sum(),
    }])

    all_ttests.to_csv(config.DESC_DIR / "GroupComparisons_AllTTests.csv", index=False)
    sex_test.to_csv(config.DESC_DIR / "GroupComparisons_Sex.csv", index=False)

    mo.ui.tabs({
        "t-tests": bu.show_table(mo, all_ttests),
        "Sex χ²": mo.vstack([
            bu.show_table(mo, sex_test),
            bu.show_table(mo, sex_contingency, "Observed counts (rows = sex, columns = birth group)"),
            bu.show_table(
                mo,
                pd.DataFrame(_expected, index=sex_contingency.index, columns=sex_contingency.columns),
                "Expected counts",
            ),
        ]),
    })
    return


@app.cell
def _(bu, config, data, mo):
    normality = bu.shapiro_table(data, ["Age_at_scan2", "tz", "gew", "TIV_harmonized", "ses0y2", "wie_gesiq"])
    normality.to_csv(config.DESC_DIR / "Normality_ShapiroWilk.csv", index=False)

    bu.show_table(mo, normality, "Shapiro-Wilk normality")
    return


@app.cell
def _(bu, config, data, ft, mo, plt, sns, vpt):
    _test = bu.two_sample_ttest(
        data.loc[vpt, "wie_gesiq"], data.loc[ft, "wie_gesiq"],
        "VP/VLBW vs Full-term", "wie_gesiq", "VP/VLBW", "Full-term",
    )

    _plot_data = data[["blsgroup", "wie_gesiq"]].dropna()
    _plot_data["Birth group"] = _plot_data["blsgroup"].map({1: "VP/VLBW", 2: "Full-term"})

    fig_iq, _ax = plt.subplots(figsize=(85 * bu.MM_TO_INCH, 85 * bu.MM_TO_INCH))

    sns.barplot(
        data=_plot_data, x="Birth group", y="wie_gesiq", order=["VP/VLBW", "Full-term"],
        errorbar="sd", capsize=0.20, color="0.82", edgecolor="black", linewidth=1, ax=_ax,
    )
    sns.stripplot(
        data=_plot_data, x="Birth group", y="wie_gesiq", order=["VP/VLBW", "Full-term"],
        color="black", alpha=0.45, size=3, jitter=0.18, ax=_ax,
    )

    _ax.set_ylim(60, 140)
    _ax.text(0.5, 140 - 80 * 0.08, bu.p_to_stars(_test["p"]), ha="center", va="center", fontsize=12, fontweight="bold")
    _ax.set_ylabel("Full-scale IQ")
    fig_iq.tight_layout()

    bu.save_figure(fig_iq, config.DESC_DIR / "FullScaleIQ_by_BirthGroup.png")

    mo.vstack([
        fig_iq,
        mo.md(
            f"FSIQ: Student t({_test['df']:.0f}) = {_test['t']:.2f}, p = {_test['p']:.4g} "
            f"(Welch t({_test['df_welch']:.1f}) = {_test['t_welch']:.2f}, p = {_test['p_welch']:.4g}); experimental unit = participant "
            f"(VP/VLBW n = {_test['Group_1_n']}, Full-term n = {_test['Group_2_n']})."
        ),
    ])
    return


if __name__ == "__main__":
    app.run()
