"""
Paths and switches shared by every notebook.

Choose the data with the BC_DATA_MODE environment variable:

    synthetic  (default)  fake data in ./synthetic_data, made by make_synthetic_data.py
    real                  the real data (read only): ../mod_analysis_final_bc/data, or the folder
                          set in BC_REAL_DATA_DIR. Not public; available only to authorized users.

All outputs go to ./outputs/<mode>/. Nothing is ever written to the original folder.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ORIGINAL_DIR = PROJECT_DIR.parent / "mod_analysis_final_bc"
# Real input data: the original analysis folder, or any folder given by BC_REAL_DATA_DIR
ORIGINAL_DATA_DIR = Path(os.environ.get("BC_REAL_DATA_DIR", ORIGINAL_DIR / "data"))

DATA_MODE = os.environ.get("BC_DATA_MODE", "synthetic").strip().lower()

if DATA_MODE not in {"synthetic", "real"}:
    raise ValueError(f"BC_DATA_MODE must be 'synthetic' or 'real', got {DATA_MODE!r}")

RAW_DIR = PROJECT_DIR / "synthetic_data" if DATA_MODE == "synthetic" else ORIGINAL_DATA_DIR
OUT_DIR = PROJECT_DIR / "outputs" / DATA_MODE

if ORIGINAL_DIR.resolve() in OUT_DIR.resolve().parents:
    raise RuntimeError("Refusing to write outputs inside the original analysis folder.")

# ------------------------------------------------------------------
# Raw inputs (same file names as the original data folder)
# ------------------------------------------------------------------
META_FILE = RAW_DIR / "BEST_Taylor_n212_descriptive_stats.xlsx"
ASEG_FILE = RAW_DIR / "aseg_volume.csv"
LH_APARC_FILE = RAW_DIR / "aparc_stats_lh_volume_DK.csv"
RH_APARC_FILE = RAW_DIR / "aparc_stats_rh_volume_DK.csv"

# ------------------------------------------------------------------
# Intermediate datasets passed between notebooks
# (same names as in the original data folder, so they can be compared)
# ------------------------------------------------------------------
EXCLUDED_FILE = OUT_DIR / "meta_aseg_aparc_subjects_excluded_correct.csv"
COMBAT_FILE = OUT_DIR / "NCH_cortical_subcortical_GMV_ROIs_correct.csv"
RESID_FILE = OUT_DIR / "NeuroCombat_resid_global_and_lobes_correct.csv"

# ------------------------------------------------------------------
# Output folders (mirror data/analysis/... in the original folder)
# ------------------------------------------------------------------
ANALYSIS_DIR = OUT_DIR / "analysis"
DESC_DIR = ANALYSIS_DIR / "descriptive_stats"
GROUP_DIR = ANALYSIS_DIR / "global_and_roi_gmv_wmv"
MOD_DIR = ANALYSIS_DIR / "moderation"
PUB_DIR = ANALYSIS_DIR / "graphs_publication"
SUPP_DIR = PUB_DIR / "supplementary_lobes"

# ------------------------------------------------------------------
# PROCESS v4.3 for R (Hayes)
# ------------------------------------------------------------------
PROCESS_R_FILE = Path(os.environ.get(
    "BC_PROCESS_R",
    Path.home() / "Documents" / "Documents - Taylor’s Laptop" / "Documents_Taylor_Laptop"
    / "sw" / "processv43" / "PROCESS v4.3 for R" / "process.R",
))
PROCESS_BOOT = int(os.environ.get("BC_PROCESS_BOOT", 5000))
PROCESS_SEED = 1234


def ensure_output_dirs():
    for folder in [
        OUT_DIR,
        DESC_DIR,
        GROUP_DIR / "graphs",
        GROUP_DIR / "tables",
        MOD_DIR / "graphs",
        MOD_DIR / "tables",
        SUPP_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)
