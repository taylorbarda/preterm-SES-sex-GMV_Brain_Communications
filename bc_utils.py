"""
Shared constants, statistics, PROCESS and plotting helpers.

Used by all notebooks, so each helper is defined in one place.
"""

import os
import re
import subprocess
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

import config

# Adding many columns one at a time is fine at this sample size
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Figures shown in marimo include legends placed outside the axes, like the saved files
plt.rcParams["savefig.bbox"] = "tight"

# ============================================================
# Figure sizing (journal page: 170 mm x 210 mm)
# ============================================================
MM_TO_INCH = 1 / 25.4
JOURNAL_WIDTH = 170 * MM_TO_INCH
JOURNAL_HEIGHT = 210 * MM_TO_INCH

# ============================================================
# Labels and palettes
# ============================================================
BIRTHGROUP_LABELS = {1: "VP/VLBW", 2: "FT"}
BIRTHGROUP_ORDER = ["VP/VLBW", "FT"]
BIRTHGROUP_PALETTE = {"VP/VLBW": "#66CDAA", "FT": "coral"}

# Publication figures (PROCESS line plots)
SES_LABELS = {1: "High SES", 2: "Middle SES", 3: "Low SES"}
SES_ORDER = ["High SES", "Middle SES", "Low SES"]
SES_PALETTE = {"High SES": "blue", "Middle SES": "orange", "Low SES": "green"}

SCANNER_STAGE_PALETTE = {"Before": "#4C72B0", "After": "#DD8452"}

# ============================================================
# Regions of interest
# ============================================================
CORTICAL_REGIONS = [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal", "lingual",
    "medialorbitofrontal", "middletemporal", "parahippocampal", "paracentral",
    "parsopercularis", "parsorbitalis", "parstriangularis", "pericalcarine",
    "postcentral", "posteriorcingulate", "precentral", "precuneus",
    "rostralanteriorcingulate", "rostralmiddlefrontal", "superiorfrontal",
    "superiorparietal", "superiortemporal", "supramarginal", "frontalpole",
    "temporalpole", "transversetemporal", "insula",
]

CORTICAL_ROIS = [
    f"{hemi}_{region}_volume"
    for hemi in ["lh", "rh"]
    for region in CORTICAL_REGIONS
]

SUBCORTICAL_ROIS = [
    "Left-Thalamus", "Right-Thalamus",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area",
    "Brain-Stem",
]

DK_LOBE_REGIONS = {
    "frontal": [
        "caudalmiddlefrontal", "rostralmiddlefrontal", "superiorfrontal",
        "lateralorbitofrontal", "medialorbitofrontal", "parsopercularis",
        "parsorbitalis", "parstriangularis", "precentral", "paracentral",
        "frontalpole",
    ],
    "parietal": [
        "inferiorparietal", "superiorparietal", "postcentral", "precuneus",
        "supramarginal",
    ],
    "temporal": [
        "bankssts", "entorhinal", "fusiform", "inferiortemporal",
        "middletemporal", "superiortemporal", "parahippocampal",
        "temporalpole", "transversetemporal",
    ],
    "occipital": ["cuneus", "lateraloccipital", "lingual", "pericalcarine"],
}

DK_LOBE_COLS = {
    lobe: [f"{hemi}_{region}_volume" for hemi in ["lh", "rh"] for region in regions]
    for lobe, regions in DK_LOBE_REGIONS.items()
}

DK_LOBE_GMV_COLS = [f"{lobe}_DK_GMV" for lobe in DK_LOBE_REGIONS]


def lobe_label(lobe_col):
    """'frontal_DK_GMV' -> 'Frontal'"""
    return lobe_col.replace("_DK_GMV", "").capitalize()


def add_lobe_composites(data):
    """Sums DK regions into frontal/parietal/temporal/occipital GMV columns."""
    data = data.copy()

    for lobe, cols in DK_LOBE_COLS.items():
        missing = [col for col in cols if col not in data.columns]
        if missing:
            raise ValueError(f"Missing DK columns for {lobe}: {missing}")

        data[f"{lobe}_DK_GMV"] = data[cols].sum(axis=1, min_count=len(cols))

    return data


def require_columns(data, cols, where=""):
    missing = [col for col in cols if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {where}: {missing}")


# ============================================================
# Statistics
# ============================================================
def format_p(p):
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}"


def p_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def run_ancova(data, formula, effect_name="C(blsgroup)"):
    """
    Type II ANCOVA (statsmodels) with partial eta squared and Cohen's f squared.
    """
    model = smf.ols(formula, data=data).fit()
    table = sm.stats.anova_lm(model, typ=2)

    residual_ss = table.loc["Residual", "sum_sq"]
    table["partial_eta_squared"] = table["sum_sq"] / (table["sum_sq"] + residual_ss)
    table["cohen_f_squared"] = table["partial_eta_squared"] / (1 - table["partial_eta_squared"])

    effect_row = table.loc[effect_name]

    return {
        "formula": formula,
        "effect": effect_name,
        "F": effect_row["F"],
        "p": effect_row["PR(>F)"],
        "df_effect": effect_row["df"],
        "df_resid": table.loc["Residual", "df"],
        "partial_eta_squared": effect_row["partial_eta_squared"],
        "cohen_f_squared": effect_row["cohen_f_squared"],
        "n": int(model.nobs),
        "model": model,
        "table": table,
    }


def ancova_label(result, sep=", "):
    return (
        f"F({result['df_effect']:.0f}{sep}{result['df_resid']:.0f}) = "
        f"{result['F']:.2f}, {format_p(result['p'])}"
    )


def two_sample_ttest(x, y, comparison, variable, group_1, group_2):
    """
    Independent-samples t-test, returned as one table row.

    t / df / p           Student's t-test (equal variances), as reported in the manuscript
    t_welch / df_welch / p_welch   Welch's t-test (unequal variances), for reference
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()

    # groups that are too small (possible in synthetic data) give NaN without a warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=getattr(stats, "SmallSampleWarning", RuntimeWarning))
        student = stats.ttest_ind(x, y, equal_var=True)
        welch = stats.ttest_ind(x, y, equal_var=False)

    return {
        "Comparison": comparison,
        "Variable": variable,
        "Test": "Student t-test (Welch alongside)",
        "Group_1": group_1,
        "Group_1_n": len(x),
        "Group_1_mean": x.mean(),
        "Group_1_SD": x.std(),
        "Group_2": group_2,
        "Group_2_n": len(y),
        "Group_2_mean": y.mean(),
        "Group_2_SD": y.std(),
        "t": student.statistic,
        "df": student.df,
        "p": student.pvalue,
        "t_welch": welch.statistic,
        "df_welch": welch.df,
        "p_welch": welch.pvalue,
    }


def shapiro_table(data, variables):
    rows = []

    for var in variables:
        if var not in data.columns:
            continue

        values = pd.to_numeric(data[var], errors="coerce").dropna()
        w, p = stats.shapiro(values)
        rows.append({"Variable": var, "Test": "Shapiro-Wilk", "W": w, "p": p, "n": len(values)})

    return pd.DataFrame(rows)


def zscore(series):
    return (series - series.mean()) / series.std()


# ============================================================
# PROCESS (Hayes) via Rscript
# ============================================================
RUNNER_R = Path(__file__).resolve().parent / "run_process.R"


def process_job(data, y, x, output_name, model, cov, z=None, complete_cases=False):
    """One PROCESS call. `data` is a key of the data dict given to run_process_jobs."""
    return {
        "data": data,
        "y": y,
        "x": x,
        "w": "ses0y2",
        "z": z or "",
        "cov": ";".join(cov),
        "model": model,
        "output_name": output_name,
        "complete_cases": complete_cases,
    }


def run_process_jobs(jobs, datasets, out_dir, process_r=None, boot=None, seed=None):
    """
    Writes each dataset to CSV, then runs every PROCESS job in one Rscript call.
    Each job's PROCESS output is written to out_dir/<output_name>.
    Returns the job log as a DataFrame.
    """
    process_r = Path(process_r or config.PROCESS_R_FILE)
    boot = boot or config.PROCESS_BOOT
    seed = seed or config.PROCESS_SEED

    if not process_r.exists():
        raise FileNotFoundError(
            f"PROCESS for R not found at:\n{process_r}\n"
            "Set the BC_PROCESS_R environment variable to the location of process.R."
        )

    out_dir = Path(out_dir)
    work_dir = out_dir / "_process_input"
    work_dir.mkdir(parents=True, exist_ok=True)

    for key, frame in datasets.items():
        frame.to_csv(work_dir / f"{key}.csv", index=False)

    jobs_file = work_dir / "jobs.csv"
    pd.DataFrame(jobs).to_csv(jobs_file, index=False)

    result = subprocess.run(
        [
            "Rscript", str(RUNNER_R),
            str(process_r), str(jobs_file), str(work_dir), str(out_dir),
            str(boot), str(seed),
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Rscript failed:\n{result.stdout}\n{result.stderr}")

    return pd.read_csv(work_dir / "jobs_log.csv")


def parse_process_plot_data(process_txt_file):
    """
    Extracts the 'Data for visualizing the conditional effect of the focal predictor'
    table from a PROCESS R text output (falls back to SPSS-style DATA LIST blocks).
    """
    process_txt_file = Path(process_txt_file)
    text = process_txt_file.read_text(encoding="utf-8", errors="ignore")

    marker = "Data for visualizing the conditional effect of the focal predictor:"

    if marker in text:
        section = text.split(marker)[-1]

        for stop_marker in [
            "******************** ANALYSIS NOTES AND ERRORS",
            "********************* ANALYSIS NOTES AND ERRORS",
            "Level of confidence",
        ]:
            section = section.split(stop_marker)[0]

        lines = [line.strip() for line in section.strip().splitlines() if line.strip()]
        var_names = re.split(r"\s+", lines[0])
        rows = [
            values for values in (re.split(r"\s+", line) for line in lines[1:])
            if len(values) == len(var_names)
        ]

        if not rows:
            raise ValueError(f"PROCESS visualization section has no rows in {process_txt_file.name}")
    else:
        matches = re.findall(
            r"DATA LIST FREE\s*/\s*(.*?)\.\s*BEGIN DATA\.\s*(.*?)\s*END DATA\.",
            text,
            re.IGNORECASE | re.DOTALL,
        )

        if not matches:
            raise ValueError(f"No PROCESS plot data block found in {process_txt_file.name}")

        var_block, data_block = matches[-1]
        var_names = [var for var in re.split(r"\s+", var_block.strip()) if var]
        rows = [re.split(r"\s+", line.strip()) for line in data_block.strip().splitlines() if line.strip()]

    plot_df = pd.DataFrame(rows, columns=var_names).apply(pd.to_numeric, errors="coerce")
    plot_df["source_file"] = process_txt_file.name

    return plot_df


def process_outcome_col(plot_df):
    """The outcome is the last column of the PROCESS plot table."""
    return [
        col for col in plot_df.columns
        if col not in ["blsgroup", "tz", "ses0y2", "sex", "source_file"]
    ][-1]


def extract_process_model_term(process_txt_file, term="Int_4"):
    """
    Returns one coefficient row from the PROCESS 'Model:' table.
    For model 3, Int_4 = X x W x Z (birth group x SES x sex).
    """
    text = Path(process_txt_file).read_text(encoding="utf-8", errors="ignore")

    if "Model:" not in text or "Product terms key:" not in text:
        return None

    model_section = text.split("Model:", 1)[1].split("Product terms key:", 1)[0]

    for line in model_section.splitlines():
        line = line.strip()

        if line.startswith(term):
            values = re.split(r"\s+", line)

            if len(values) < 6:
                return None

            return {
                "term": values[0],
                "coeff": float(values[1]),
                "se": float(values[2]),
                "t": float(values[3]),
                "p": float(values[4]),
                "LLCI": float(values[5]),
            }

    return None


# ============================================================
# Plotting
# ============================================================
def save_figure(fig, path, dpi=300, tiff=False):
    """Saves a PNG (and optionally a TIFF next to it) and returns the figure."""
    path = Path(path)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")

    if tiff:
        fig.savefig(path.with_suffix(".tiff"), dpi=dpi, format="tiff", bbox_inches="tight")

    return fig


def birthgroup_panel(
    ax,
    data,
    y_col,
    y_label,
    result,
    panel_label=None,
    ylim=None,
    yticks=None,
    zero_line=False,
    box_linewidth=1,
    point_size=3,
    point_alpha=0.55,
    stat_fontsize=9,
    stat_height=0.07,
    stat_sep=", ",
):
    """Box + strip plot of one outcome by birth group, annotated with the ANCOVA result."""
    plot_data = data[["birth_group", y_col, "MRT_ID"]].dropna()

    sns.boxplot(
        data=plot_data, x="birth_group", y=y_col, hue="birth_group",
        order=BIRTHGROUP_ORDER, palette=BIRTHGROUP_PALETTE, legend=False,
        showfliers=False, linewidth=box_linewidth, ax=ax,
    )
    sns.stripplot(
        data=plot_data, x="birth_group", y=y_col, order=BIRTHGROUP_ORDER,
        color="black", alpha=point_alpha, jitter=0.18, size=point_size, ax=ax,
    )

    if zero_line:
        ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    if ylim is not None:
        ax.set_ylim(ylim)
    if yticks is not None:
        ax.set_yticks(yticks)

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=False))
    ax.ticklabel_format(style="plain", axis="y")

    y_min, y_max = ax.get_ylim()
    ax.text(
        0.5, y_max - (y_max - y_min) * stat_height, ancova_label(result, sep=stat_sep),
        ha="center", va="center", fontsize=stat_fontsize,
    )

    ax.set_xlabel("Birth group")
    ax.set_ylabel(y_label)
    ax.set_title(panel_label or "", loc="left", fontweight="bold")


def plot_process_ses(
    ax,
    plot_df,
    x_col,
    y_col,
    title,
    y_label,
    panel_label=None,
    y_scale=1000,
    ylim=None,
):
    """
    Plots PROCESS conditional-effect lines (one per SES level).
    x_col is 'blsgroup' (birth group) or 'tz' (gestational age).
    """
    plot_data = plot_df.copy()
    plot_data["SES_label"] = plot_data["ses0y2"].map(SES_LABELS)
    plot_data["plot_y"] = plot_data[y_col] / y_scale

    if x_col == "blsgroup":
        plot_data["BirthGroup"] = pd.Categorical(
            plot_data["blsgroup"].map(BIRTHGROUP_LABELS),
            categories=BIRTHGROUP_ORDER,
            ordered=True,
        )
        x_plot, x_label = "BirthGroup", "Birth Group"
    else:
        x_plot, x_label = x_col, "Gestational Age (weeks)"

    sns.lineplot(
        data=plot_data, x=x_plot, y="plot_y", hue="SES_label",
        hue_order=SES_ORDER, palette=SES_PALETTE, marker="o",
        markersize=3.5, linewidth=1.1, errorbar=None, ax=ax,
    )

    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_title(title, fontsize=9.5, pad=6)
    ax.set_xlabel(x_label, fontsize=8.5)
    ax.set_ylabel(y_label, fontsize=8.5)
    ax.tick_params(axis="both", labelsize=7.5)
    ax.grid(True, linewidth=0.45, alpha=0.55)

    if panel_label is not None:
        ax.text(
            -0.16, 1.08, panel_label, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top",
        )

    if ax.legend_ is not None:
        ax.legend_.remove()


def restyle_process_axes(
    axes,
    tick_size=12,
    label_size=12,
    title_size=12,
    panel_size=14,
    panel_pos=(-0.25, 1.04),
    publication=True,
):
    """
    Final styling of PROCESS panels.
    publication=True: lighter split grid, no top/right spines, thicker lines.
    """
    for ax in np.ravel(axes):
        ax.tick_params(axis="both", labelsize=tick_size)
        ax.xaxis.label.set_size(label_size)
        ax.yaxis.label.set_size(label_size)
        ax.title.set_fontsize(title_size)

        for text in ax.texts:
            text.set_fontsize(panel_size)
            text.set_fontweight("bold")
            text.set_position(panel_pos)

        if publication:
            ax.title.set_y(1.03)
            ax.grid(True, axis="y", linewidth=0.6, alpha=0.35)
            ax.grid(True, axis="x", linewidth=0.4, alpha=0.20)

            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

            for line in ax.lines:
                line.set_linewidth(1.6)
                line.set_markersize(4.5)
        else:
            ax.grid(True, linewidth=0.45, alpha=0.55)


def ses_legend(fig, bbox_to_anchor, fontsize=12, title_fontsize=12, linewidth=1.6, markersize=4.5):
    handles = [
        plt.Line2D([0], [0], color=SES_PALETTE[label], marker="o",
                   linewidth=linewidth, markersize=markersize, label=label)
        for label in SES_ORDER
    ]

    fig.legend(
        handles=handles, title="SES Level", loc="upper center", ncol=3,
        frameon=False, fontsize=fontsize, title_fontsize=title_fontsize,
        bbox_to_anchor=bbox_to_anchor,
    )


def scanner_effect_barplot(results, label_col, title=None, figsize=None, max_bars=None, output_file=None, dpi=300):
    """Horizontal bars of scanner F before vs after NeuroCombat, sorted by 'Before' F."""
    order = (
        results[results["Stage"] == "Before"]
        .sort_values("F", ascending=False)[label_col]
        .tolist()
    )

    if max_bars is not None:
        order = order[:max_bars]

    plot_data = results[results[label_col].isin(order)]

    if figsize is None:
        figsize = (JOURNAL_WIDTH, JOURNAL_HEIGHT if len(order) > 24 else 150 * MM_TO_INCH)

    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(
        data=plot_data, y=label_col, x="F", hue="Stage", order=order,
        hue_order=["Before", "After"], palette=SCANNER_STAGE_PALETTE,
        edgecolor="white", linewidth=0.5, ax=ax,
    )

    ax.set_title(title or "", fontsize=12, pad=7)
    ax.set_xlabel("ANCOVA F-statistic for scanner", fontsize=12)
    ax.set_ylabel("")
    ax.legend(title="", loc="lower right", frameon=True)
    ax.grid(axis="x", linewidth=0.6, alpha=0.55)
    ax.grid(axis="y", visible=False)
    sns.despine(ax=ax)
    fig.tight_layout()

    if output_file is not None:
        save_figure(fig, output_file, dpi=dpi)

    return fig


# ============================================================
# marimo display helpers
# ============================================================
def mode_banner(mo):
    """Callout at the top of each notebook saying which data is being used."""
    if config.DATA_MODE == "synthetic":
        return mo.callout(
            mo.md(
                "**Data mode: `synthetic`** (fake data, results are meaningless).  \n"
                f"Outputs → `outputs/synthetic/`. Run with `BC_DATA_MODE=real` (authorized users only) to use the real data."
            ),
            kind="warn",
        )

    return mo.callout(
        mo.md(
            f"**Data mode: `real`**, reading (read-only) from `{config.RAW_DIR}`.  \n"
            "Outputs → `outputs/real/`."
        ),
        kind="info",
    )


def show_table(mo, df, label=None):
    if df.index.name or isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = df.rename(columns=str)
    table = mo.ui.table(df, selection=None, page_size=min(max(len(df), 1), 100))
    return mo.vstack([mo.md(f"**{label}**"), table]) if label else table
