"""
Paths and switches shared by every notebook.

Choose the data with the BC_DATA_MODE environment variable:

    synthetic  (default)  fake data in ./synthetic_data, made by make_synthetic_data.py
    real                  the real data (read only) in the folder set by BC_REAL_DATA_DIR.
                          Not public; available only to authorized users.

All outputs go to ./outputs/<mode>/. The input folder is only ever read.
"""

import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATA_MODE = os.environ.get("BC_DATA_MODE", "synthetic").strip().lower()

if DATA_MODE not in {"synthetic", "real"}:
    raise ValueError(f"BC_DATA_MODE must be 'synthetic' or 'real', got {DATA_MODE!r}")

if DATA_MODE == "real":
    if "BC_REAL_DATA_DIR" not in os.environ:
        raise RuntimeError("BC_DATA_MODE=real needs BC_REAL_DATA_DIR=/path/to/the/real/input/files")
    RAW_DIR = Path(os.environ["BC_REAL_DATA_DIR"])
else:
    RAW_DIR = PROJECT_DIR / "synthetic_data"
OUT_DIR = PROJECT_DIR / "outputs" / DATA_MODE

if RAW_DIR.resolve() in OUT_DIR.resolve().parents:
    raise RuntimeError("Refusing to write outputs inside the input data folder.")

# ------------------------------------------------------------------
# Raw inputs
# ------------------------------------------------------------------
META_FILE = RAW_DIR / "BEST_Taylor_n212_descriptive_stats.xlsx"
ASEG_FILE = RAW_DIR / "aseg_volume.csv"
LH_APARC_FILE = RAW_DIR / "aparc_stats_lh_volume_DK.csv"
RH_APARC_FILE = RAW_DIR / "aparc_stats_rh_volume_DK.csv"

# ------------------------------------------------------------------
# Intermediate datasets passed between notebooks
# ------------------------------------------------------------------
EXCLUDED_FILE = OUT_DIR / "meta_aseg_aparc_subjects_excluded_correct.csv"
COMBAT_FILE = OUT_DIR / "NCH_cortical_subcortical_GMV_ROIs_correct.csv"
RESID_FILE = OUT_DIR / "NeuroCombat_resid_global_and_lobes_correct.csv"

# ------------------------------------------------------------------
# Output folders
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
# Download process.R from https://www.processmacro.org and set BC_PROCESS_R,
# or place it next to this file.
PROCESS_R_FILE = Path(os.environ.get("BC_PROCESS_R", PROJECT_DIR / "process.R"))
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
