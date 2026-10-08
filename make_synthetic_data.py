"""
Creates FAKE input data with the same file names, formats and columns as the real data:

    BEST_Taylor_n212_descriptive_stats.xlsx   participant metadata
    aseg_volume.csv                           FreeSurfer aseg table (tab separated)
    aparc_stats_lh_volume_DK.csv              FreeSurfer DK volumes, left (tab separated)
    aparc_stats_rh_volume_DK.csv              FreeSurfer DK volumes, right (tab separated)

The values are random draws from plausible ranges, NOT derived from the real participants.
Built-in effects (scanner offsets, a few outliers, a preterm GMV deficit that depends on
SES and sex) make sure every step of the pipeline has something to find.

Run:  .venv/bin/python make_synthetic_data.py
"""

import numpy as np
import pandas as pd

import bc_utils as bu
from config import PROJECT_DIR

SEED = 20261006
N_META = 212          # participants in the metadata file
N_EXTRA_SCANS = 12    # scans with no metadata row (dropped by the inner merge)
N_OUTLIERS = 4        # participants with implausible GMV/WMV

OUT_DIR = PROJECT_DIR / "synthetic_data"

# Approximate adult DK cortical volumes (mm3), one hemisphere
CORTICAL_BASE = {
    "bankssts": 2600, "caudalanteriorcingulate": 2100, "caudalmiddlefrontal": 6500,
    "cuneus": 3300, "entorhinal": 2000, "fusiform": 9800, "inferiorparietal": 13000,
    "inferiortemporal": 11000, "isthmuscingulate": 2600, "lateraloccipital": 12500,
    "lateralorbitofrontal": 8200, "lingual": 7500, "medialorbitofrontal": 5600,
    "middletemporal": 11500, "parahippocampal": 2100, "paracentral": 3800,
    "parsopercularis": 4600, "parsorbitalis": 2400, "parstriangularis": 3900,
    "pericalcarine": 2000, "postcentral": 10500, "posteriorcingulate": 3300,
    "precentral": 14000, "precuneus": 10000, "rostralanteriorcingulate": 2500,
    "rostralmiddlefrontal": 17000, "superiorfrontal": 23000, "superiorparietal": 13500,
    "superiortemporal": 12500, "supramarginal": 11000, "frontalpole": 900,
    "temporalpole": 2300, "transversetemporal": 1200, "insula": 7200,
}

SUBCORTICAL_BASE = {
    "Thalamus": 7800, "Caudate": 3700, "Putamen": 5200, "Pallidum": 2000,
    "Hippocampus": 4300, "Amygdala": 1750, "Accumbens-area": 600,
}

SCANNER_PROBS = [0.08, 0.24, 0.58, 0.10]          # Scanner_ID 1-4
SCANNER_SHIFT = {1: -0.03, 2: 0.02, 3: 0.0, 4: 0.04}  # multiplicative site effect


def make_metadata(rng, n):
    blsgroup = rng.choice([1, 2], size=n, p=[0.48, 0.52])
    preterm = blsgroup == 1

    tz = np.where(preterm, rng.integers(25, 32, n), rng.integers(37, 43, n))
    gew = np.where(
        preterm,
        rng.normal(1150, 280, n).clip(550, 1490),
        rng.normal(3400, 450, n).clip(2500, 4700),
    ).round()

    inti = np.where(preterm, rng.normal(11, 4, n).clip(0, 18).round(2), np.nan)
    iq = rng.normal(np.where(preterm, 94, 103), 13).round()
    iq[rng.random(n) < 0.03] = np.nan

    centres = rng.choice(["BN", "MUC"], size=n)

    return pd.DataFrame({
        "MRT_ID": [f"SYN-{centre}-{i:03d}" for i, centre in enumerate(centres, start=1)],
        "blsgroup": blsgroup,
        "tz": tz,
        "gew": gew.astype(int),
        "sex": rng.choice([1, 2], size=n, p=[0.58, 0.42]),
        "Age_at_scan2": rng.normal(26.8, 0.65, n).clip(25.4, 29.0),
        "Scanner_ID": rng.choice([1, 2, 3, 4], size=n, p=SCANNER_PROBS),
        "ses0y2": rng.choice([1, 2, 3], size=n, p=[0.30, 0.45, 0.25]),
        "inti": inti,
        "wie_gesiq": iq,
    })


def make_volumes(rng, meta):
    n = len(meta)
    male = (meta["sex"].to_numpy() == 1).astype(float)
    preterm = (meta["blsgroup"].to_numpy() == 1).astype(float)
    ses = meta["ses0y2"].to_numpy()
    scanner = meta["Scanner_ID"].map(SCANNER_SHIFT).to_numpy()

    tiv = rng.normal(1.45e6, 1.1e5, n) + male * 1.4e5 - preterm * 7e4

    # Preterm GMV deficit, larger with lower SES, mostly in females (a 3-way interaction)
    gmv_effect = -preterm * (6000 + 5000 * (ses - 1) * np.where(male == 1, 0.3, 1.0))
    gmv = (0.44 * tiv + gmv_effect + rng.normal(0, 18000, n)) * (1 + scanner)
    wmv = (0.33 * tiv - preterm * 15000 + rng.normal(0, 15000, n)) * (1 + scanner)

    outliers = rng.choice(n, size=N_OUTLIERS, replace=False)
    gmv[outliers[:2]] -= 150000
    wmv[outliers[2:]] += 140000

    gmv_scale = gmv / gmv.mean()

    aseg = pd.DataFrame({"Measure:volume": meta["MRT_ID"]})

    for hemi in ["Left", "Right"]:
        for region, base in SUBCORTICAL_BASE.items():
            aseg[f"{hemi}-{region}"] = base * gmv_scale * rng.normal(1, 0.07, n)

    aseg["Brain-Stem"] = 21000 * (tiv / tiv.mean()) * rng.normal(1, 0.07, n)
    aseg["BrainSegVolNotVent"] = gmv + wmv + rng.normal(60000, 5000, n)
    aseg["CerebralWhiteMatterVol"] = wmv
    aseg["TotalGrayVol"] = gmv
    aseg["EstimatedTotalIntraCranialVol"] = tiv

    aparc = {}

    for hemi in ["lh", "rh"]:
        table = pd.DataFrame({f"{hemi}.aparc.volume": meta["MRT_ID"]})

        for region, base in CORTICAL_BASE.items():
            table[f"{hemi}_{region}_volume"] = (
                base * gmv_scale * (1 + scanner) * rng.normal(1, 0.08, n)
            ).round()

        table["BrainSegVolNotVent"] = aseg["BrainSegVolNotVent"]
        table["eTIV"] = tiv
        aparc[hemi] = table

    return aseg, aparc


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(exist_ok=True)

    meta_all = make_metadata(rng, N_META + N_EXTRA_SCANS)
    aseg, aparc = make_volumes(rng, meta_all)

    meta = meta_all.iloc[:N_META]
    meta.to_excel(OUT_DIR / "BEST_Taylor_n212_descriptive_stats.xlsx", index=False)

    aseg.to_csv(OUT_DIR / "aseg_volume.csv", sep="\t", index=False)
    aparc["lh"].to_csv(OUT_DIR / "aparc_stats_lh_volume_DK.csv", sep="\t", index=False)
    aparc["rh"].to_csv(OUT_DIR / "aparc_stats_rh_volume_DK.csv", sep="\t", index=False)

    assert set(bu.SUBCORTICAL_ROIS) <= set(aseg.columns)
    print(f"Synthetic data written to {OUT_DIR}")
    print(f"  metadata: {meta.shape}, aseg: {aseg.shape}, aparc lh/rh: {aparc['lh'].shape}")


if __name__ == "__main__":
    main()
