# Brain_Communications_code

Analysis code for:

> **Infant socioeconomic status moderates preterm birth effects on adult gray matter in females only**
> Barda T, Thalhammer M, Schmitz-Koep B, Menegaux A, Bartmann P, Wolke D, Sorg C, Hedderich DM. *Brain Communications* (under revision).

This folder is a cleaned-up [marimo](https://marimo.io) version of the Jupyter notebooks in
`../mod_analysis_final_bc/`. **The original folder is never modified.** In real-data mode it is only read.

---

## Quick start

```bash
cd marimo_mod_analysis_final_bc

# one-time setup
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# run everything on FAKE data and build an HTML site of the results
./run_all.sh
open outputs/synthetic/site/index.html
```

Open a notebook interactively (live editing, tabs, dropdowns):

```bash
.venv/bin/marimo edit 05_moderation_process.py
```

Run on the **real data**. This reads `../mod_analysis_final_bc/data/` and writes only to `outputs/real/`:

```bash
./run_all.sh real
BC_DATA_MODE=real .venv/bin/marimo edit 07_compare_with_original.py
```

Notebook 07 checks every new table and PROCESS output against the files the original
notebooks produced, and shows each new figure next to the original one.

---

## Requirements

| | |
|---|---|
| Python | 3.12 (package versions pinned in `requirements.txt`, same as the original `.venv`) |
| R | ≥ 4.x with `Rscript` on the PATH |
| PROCESS v4.3 for R (Hayes) | Download `process.R` from [processmacro.org](https://www.processmacro.org) and set `BC_PROCESS_R=/path/to/process.R`. |

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BC_DATA_MODE` | `synthetic` | `synthetic` or `real` |
| `BC_PROCESS_R` | — | path to `process.R` |
| `BC_PROCESS_BOOT` | `5000` | PROCESS bootstrap samples |

---

## Pipeline

Run the notebooks in order. Each one saves the file the next one needs.

| Notebook | What it does | Replaces | Main output |
|---|---|---|---|
| `01_subject_exclusion.py` | Merge metadata + aseg + DK aparc; drop GMV/WMV residual outliers (> 3 SD) | `A1_subject_exclusion.ipynb` | `meta_aseg_aparc_subjects_excluded_correct.csv` |
| `02_neurocombat.py` | NeuroCombat for global volumes and 83 GMV ROIs; DK lobe sums; scanner ANCOVAs before/after (eFigures 1–2) | `A2_neurocombat_bc.ipynb` | `NCH_cortical_subcortical_GMV_ROIs_correct.csv` |
| `03_descriptive_stats.py` | Table 1: descriptives, Student's t-tests (as in the paper) with Welch's alongside, χ² for sex, Shapiro–Wilk, FSIQ plot | `B1_descriptive_stats_bc.ipynb` | `analysis/descriptive_stats/` |
| `04_group_differences.py` | TIV residualization, histograms/Q–Q plots, birth-group ANCOVAs (Fig. 1, eTable 1), DK lobe ANCOVAs + FDR (eFigure 3, eTable 2) | `B2_creating_residuals_bc.ipynb` | `NeuroCombat_resid_global_and_lobes_correct.csv`, `analysis/global_and_roi_gmv_wmv/` |
| `05_moderation_process.py` | 42 PROCESS models (model 3 whole group, model 1 by sex; birth group and GA; global + lobe outcomes): eTables 3–7 | `B3_moderation_analysis_bc.ipynb` | `analysis/moderation/*.txt` |
| `06_publication_figures.py` | Figs. 1–3, eFigures 4–7, FSIQ OLS models (eTable 8) | `C1_visualization_bc.ipynb` | `analysis/graphs_publication/`, `analysis/fsiq_gmv_ols_*.txt` |
| `07_compare_with_original.py` | Real mode only: compares all outputs with the original results | — | (on screen) |

Output file names and sub-folders are the same as in `../mod_analysis_final_bc/data/`, so
`outputs/real/<path>` can be compared directly with `../mod_analysis_final_bc/data/<path>`.

### Folder layout

```
marimo_mod_analysis_final_bc/
├── 01_ … 07_*.py            marimo notebooks
├── bc_utils.py              shared constants, statistics, PROCESS and plotting helpers
├── config.py                paths + data mode switch
├── run_process.R            runs a list of PROCESS models in one Rscript call
├── make_synthetic_data.py   creates the fake input data
├── run_all.sh               runs the pipeline and builds the HTML site
├── requirements.txt
├── synthetic_data/          fake inputs (same file names/formats as the real data)
└── outputs/
    ├── synthetic/           results from fake data (+ site/)
    └── real/                results from real data (+ site/)
```

---

## Synthetic data

`make_synthetic_data.py` writes four files with **the same names, formats and column names** as
the real inputs (`BEST_Taylor_n212_descriptive_stats.xlsx`, `aseg_volume.csv`,
`aparc_stats_{lh,rh}_volume_DK.csv`). All values are random draws from plausible ranges and
are **not derived from the participants**. Effects are built in so every step has something
to find: scanner offsets (removed by NeuroCombat), 4 outliers (removed in 01), extra scans
without metadata, and a preterm GMV deficit that depends on SES and sex.

The results from synthetic data are meaningless. They only show that the code runs from start to finish.

---

## What changed compared with the Jupyter notebooks

### Redundant code removed

- **Shared helpers in one place** (`bc_utils.py`): ROI/lobe lists, palettes, `run_ancova`,
  `format_p`, the t-test helper, Shapiro table, PROCESS parsers and plotting functions were
  defined again in several notebooks.
- **A2**: three almost identical scanner-comparison functions (lobes / global / ROIs) became
  one, as did the two pivot/summary blocks and the lobe composite code.
- **B2**: `plot_birthgroup_boxstrip` and `add_birthgroup_panel` were merged into one panel function. The 7 single plots
  are made in one loop. The commented-out ROI analysis was removed (reviewers asked for lobes instead).
- **B3**: cells 5–6 were repeated in cell 9 (removed). The eight `%%R` functions that differed only
  in X, covariates or data became **one job list** run by `run_process.R`. This also means no
  `rpy2`/`%%R` magic and no `pip install` inside the notebook.
- **C1**: `plot_process_ga_ses` and `plot_process_birthgroup_ses` were merged. The restyle and legend blocks
  that were copied into each figure became `restyle_process_axes` / `ses_legend`. The lobe
  grids were built twice (cells 24 and 25) and are now built once. A broken cell that called `fig.legend` before
  `fig` existed (cell 8) was removed.
- Empty cells and leftover ROI code were removed.

### Bugs fixed (check these when you compare with the original outputs)

1. **A2 overwrote the lobe scanner table.** `scanner_effects_file` was reassigned to
   `DK_lobe_scanner_effects_before_after_NeuroCombat.csv` in cell 4, so cell 11 then wrote the
   **ROI** results into the lobe file. Now each table goes to its own file:
   `DK_lobe_…` and `ROI_scanner_effects_before_after_NeuroCombat.csv`.
2. **C1 supplementary lobe grids**: the TIFF was saved under the `.png` file name, so the PNG
   was replaced by TIFF data. Now a real `.png` and a `.tiff` are saved.
3. **B3 exploratory interaction plots**: the SES legend overlapped the "Male"/"Female" panel
   titles. The legend is now placed above them.
4. **B3 PROCESS path** pointed to `…/Documents/Documents_Taylor_Laptop/sw/…`, which doesn't exist
   any more. It is now in `config.py` and can be changed with `BC_PROCESS_R`.

### Added (from the manuscript cross-check)

- Every two-sample t-test now reports **both** Student's (`t`, `df`, `p`, the values in the
  manuscript) and Welch's (`t_welch`, `df_welch`, `p_welch`). The original notebooks only computed
  Welch's, but the manuscript reports Student's. The high-SES female test is also saved as
  `high_SES_female_GMV_resid_ttest.csv`.

- Histograms and Q–Q plots of TIV-normalized GMV/WMV (Methods 2.6) → notebook 04.
- Full-sample OLS `FSIQ ~ GMV × birth group + age + sex` (**eTable 8**) → notebook 06.
  This model was not in any of the original notebooks.
- VP/VLBW OLS `FSIQ ~ GMV + age + sex` with one-tailed p, as described in the manuscript. The existing
  `fsiq_gmv_ols_preterms.txt` was made in R with **age only**, so notebook 06 writes both versions to
  `fsiq_gmv_ols_preterms.txt`.
- Lists of significant lobes (`DK_lobes_significant_birthgroup_{FDR,uncorrected}.csv/.txt`).

### Not reproduced

- ROI-level analysis files (`GMV_ROI_*`, `ROI_*_sig_fdr.txt`, `graphs_publication/supplementary_roi/`). They came from
  the ROI analysis that was replaced by the lobe analysis.
- Older figure versions that the current notebooks no longer make:
  `Figure2_GMV_birthgroup_SES_interactions.png`, `Figure_GMV_gestational_age_SES_interactions.png`,
  `Figure_WMV_TIV_gestational_age_SES_interactions.png`, `PROCESS_WMV_TIV_gestational_age_SES_interactions.png` (non-`1legend`),
  `Figure_moderated_moderation_GMV_birthgroup_SES_sex_updated.png` (older than the final Fig. 2), and
  B2's `Combined_GMV_WMV_TIV_birthgroup_figure.png` (same content as Figure 1).
- `NeuroCombat_resid_global_and_roi_correct.csv`: B2 saved this, but nothing read it. The lobe
  version contains the same data plus the lobe columns.

---

## Data privacy

The participant data are confidential and are **never** part of this repository:

- `.gitignore` excludes `outputs/real/`, any `data/` folder, data files outside `synthetic_data/`,
  marimo's `__marimo__/` session caches (they store displayed results), and private notes.
- Only the code, the synthetic data, and results computed from synthetic data are published.
- To reproduce the real analysis, request data access from the corresponding author, then run
  `./run_all.sh real` locally.
