# Brain_Communications_code

Analysis code for:

> **Infant socioeconomic status moderates preterm birth effects on adult gray matter in females only**
> Barda T, Thalhammer M, Schmitz-Koep B, Menegaux A, Bartmann P, Wolke D, Sorg C, Hedderich DM.
> *Brain Communications*, 2026. DOI: *to be added*

The analysis is written as [marimo](https://marimo.io) notebooks. The participant data
(Bavarian Longitudinal Study) are confidential and are **not** included. The repository contains
synthetic (fake) data with the same structure, so anyone can run the full pipeline and check that
the code works.

---

## Quick start

```bash
git clone https://github.com/taylorbarda/Brain_Communications_code.git
cd Brain_Communications_code

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

### Real data (authorized users only)

The participant data are not public. Researchers granted data access by the corresponding
author can place the four input files (same names as in `synthetic_data/`) in a local folder and run:

```bash
BC_REAL_DATA_DIR=/path/to/data ./run_all.sh real
```

Real-data results are written to `outputs/real/`, which is excluded from version control.

---

## Requirements

| | |
|---|---|
| Python | 3.12 (package versions pinned in `requirements.txt`) |
| R | ≥ 4.x with `Rscript` on the PATH |
| PROCESS v4.3 for R (Hayes) | Download `process.R` from [processmacro.org](https://www.processmacro.org) and set `BC_PROCESS_R=/path/to/process.R` (or place it in this folder). PROCESS is not included here because it has its own license. |

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `BC_DATA_MODE` | `synthetic` | `synthetic` or `real` (real requires data access) |
| `BC_REAL_DATA_DIR` | — | folder with the real input files (authorized users only) |
| `BC_PROCESS_R` | `./process.R` | path to `process.R` |
| `BC_PROCESS_BOOT` | `5000` | PROCESS bootstrap samples |

---

## Pipeline

Run the notebooks in order (`run_all.sh` does this). Each one saves the file the next one needs.

| Notebook | What it does | Main output |
|---|---|---|
| `01_subject_exclusion.py` | Merge metadata, FreeSurfer aseg and Desikan–Killiany aparc volumes; drop GMV/WMV residual outliers (> 3 SD) | `meta_aseg_aparc_subjects_excluded_correct.csv` |
| `02_neurocombat.py` | ComBat harmonisation of global volumes and 83 GMV regions across scanners; lobe sums; scanner ANCOVAs before/after (eFigures 1–2) | `NCH_cortical_subcortical_GMV_ROIs_correct.csv` |
| `03_descriptive_stats.py` | Table 1: descriptives, Student's and Welch's t-tests, χ² for sex, Shapiro–Wilk, FSIQ plot | `analysis/descriptive_stats/` |
| `04_group_differences.py` | TIV residualisation, histograms/Q–Q plots, birth-group ANCOVAs (Fig. 1, eTable 1), lobe ANCOVAs with FDR (eFigure 3, eTable 2) | `NeuroCombat_resid_global_and_lobes_correct.csv`, `analysis/global_and_roi_gmv_wmv/` |
| `05_moderation_process.py` | 42 PROCESS models (model 3 whole group, model 1 by sex; birth group and gestational age; global and lobe outcomes): eTables 3–7 | `analysis/moderation/*.txt` |
| `06_publication_figures.py` | Figs. 1–3, eFigures 4–7, FSIQ OLS models (eTable 8) | `analysis/graphs_publication/`, `analysis/fsiq_gmv_ols_*.txt` |

### Folder layout

```
Brain_Communications_code/
├── 01_ … 06_*.py            marimo notebooks
├── bc_utils.py              shared constants, statistics, PROCESS and plotting helpers
├── config.py                paths + data mode switch
├── run_process.R            runs a list of PROCESS models in one Rscript call
├── make_synthetic_data.py   creates the fake input data
├── run_all.sh               runs the pipeline and builds the HTML site
├── requirements.txt
├── synthetic_data/          fake inputs (same file names/formats as the real data)
└── outputs/
    ├── synthetic/           results from fake data (+ site/)
    └── real/                results from real data (local only, never committed)
```

---

## Synthetic data

`make_synthetic_data.py` writes four files with **the same names, formats and column names** as
the real inputs (`BEST_Taylor_n212_descriptive_stats.xlsx`, `aseg_volume.csv`,
`aparc_stats_{lh,rh}_volume_DK.csv`). All values are random draws from plausible ranges and
are **not derived from the participants**. Effects are built in so every step has something
to find: scanner offsets (removed by ComBat), 4 outliers (removed in 01), extra scans
without metadata, and a preterm GMV deficit that depends on SES and sex.

The results from synthetic data are meaningless. They only show that the code runs from start to finish.

---

## Data privacy

The participant data are confidential and are **never** part of this repository:

- `.gitignore` excludes `outputs/real/`, any `data/` folder, data files outside `synthetic_data/`,
  and marimo's `__marimo__/` session caches (they store displayed results).
- Only the code, the synthetic data, and results computed from synthetic data are published.
- To reproduce the real analysis, request data access from the corresponding author, then run
  `./run_all.sh real` locally.

## Funding

This work was supported by the Deutsche Forschungsgemeinschaft (DFG; HE 8967/3-1 to D.M.H.).
See the paper for the full funding statement.

## License

© 2026 Taylor Barda. This code is released under the
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(CC BY-NC-SA 4.0); see `LICENSE`. You may reuse and adapt it for non-commercial purposes
with attribution, and must share adaptations under the same license.
Please cite the paper above if you use this code.
