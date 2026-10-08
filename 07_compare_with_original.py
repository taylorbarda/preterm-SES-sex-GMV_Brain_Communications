import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 07 · Compare new outputs with the original analysis

    After running notebooks 01–06 with `BC_DATA_MODE=real`, this notebook checks every result
    against the files the original Jupyter notebooks produced in `../mod_analysis_final_bc/data/`
    (read only):

    - **CSV tables**: every numeric column compared with a relative tolerance of 1e-6
    - **PROCESS outputs**: sample size and every coefficient (b, SE, t, p) in the `Model:` table
    - **Figures**: listed side by side (compare them by eye)
    """)
    return


@app.cell
def _():
    import re

    import marimo as mo
    import numpy as np
    import pandas as pd

    import bc_utils as bu
    import config
    return bu, config, mo, np, pd, re


@app.cell
def _(bu, config, mo):
    mo.stop(
        config.DATA_MODE != "real",
        mo.vstack([
            bu.mode_banner(mo),
            mo.callout(
                mo.md(
                    "This comparison only makes sense with the real data. Run the pipeline with "
                    "`BC_DATA_MODE=real` (see README), then open this notebook the same way."
                ),
                kind="neutral",
            ),
        ]),
    )
    bu.mode_banner(mo)
    return


@app.cell
def _(config):
    new_root = config.OUT_DIR
    original_root = config.ORIGINAL_DATA_DIR

    def new_files(pattern):
        return sorted(
            path for path in new_root.rglob(pattern)
            if "_process_input" not in path.parts and "site" not in path.parts
        )
    return new_files, new_root, original_root


@app.cell
def _(mo):
    mo.md(r"""
    ## CSV tables
    """)
    return


@app.cell
def _(new_files, new_root, np, original_root, pd):
    def compare_csv(new_path, original_path, rtol=1e-6, atol=1e-8):
        new = pd.read_csv(new_path)
        original = pd.read_csv(original_path)

        if list(new.columns) != list(original.columns):
            only_new = sorted(set(new.columns) - set(original.columns))
            only_original = sorted(set(original.columns) - set(new.columns))
            return "columns differ", f"only new: {only_new[:5]} · only original: {only_original[:5]}", np.nan

        if new.shape != original.shape:
            return "shape differs", f"new {new.shape} vs original {original.shape}", np.nan

        max_diff = 0.0
        for col in new.columns:
            if pd.api.types.is_numeric_dtype(new[col]) and pd.api.types.is_numeric_dtype(original[col]):
                a = new[col].to_numpy(dtype=float)
                b = original[col].to_numpy(dtype=float)
                if not np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True):
                    diff = np.nanmax(np.abs(a - b))
                    return "values differ", f"column '{col}' (max abs diff {diff:.4g})", diff
                max_diff = max(max_diff, np.nanmax(np.abs(a - b), initial=0))
            elif not new[col].astype(str).equals(original[col].astype(str)):
                return "values differ", f"text column '{col}'", np.nan

        return "match", "", max_diff

    _rows = []

    for _path in new_files("*.csv"):
        _rel = _path.relative_to(new_root)
        _original = original_root / _rel

        if not _original.exists():
            _rows.append({"file": str(_rel), "status": "not in original", "detail": "", "max_abs_diff": np.nan})
            continue

        _status, _detail, _diff = compare_csv(_path, _original)
        _rows.append({"file": str(_rel), "status": _status, "detail": _detail, "max_abs_diff": _diff})

    csv_comparison = pd.DataFrame(_rows)
    return (csv_comparison,)


@app.cell
def _(bu, csv_comparison, mo):
    _counts = csv_comparison["status"].value_counts().to_dict()
    mo.vstack([
        mo.md(" · ".join(f"**{status}**: {n}" for status, n in _counts.items())),
        bu.show_table(mo, csv_comparison),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## PROCESS outputs
    """)
    return


@app.cell
def _(config, new_root, np, original_root, pd, re):
    def parse_process_model(path):
        """Sample size and the Model: coefficient table (handles R's wrapped tables)."""
        text = path.read_text(errors="ignore")
        n_match = re.search(r"Sample size:\s*(\d+)", text)
        section = text.split("Model:", 1)[1].split("Product terms key:", 1)[0]

        table, columns = {}, []
        for line in section.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if all(re.fullmatch(r"[A-Za-z]+", tok) for tok in tokens):
                columns = tokens
                continue
            table.setdefault(tokens[0], {}).update(zip(columns, map(float, tokens[1:])))

        return (int(n_match.group(1)) if n_match else None), table

    _rows = []

    for _path in sorted(config.MOD_DIR.glob("*.txt")):
        _rel = _path.relative_to(new_root)
        _original = original_root / _rel

        if not _original.exists():
            _rows.append({"file": _path.name, "status": "not in original"})
            continue

        try:
            _n_new, _new = parse_process_model(_path)
            _n_orig, _orig = parse_process_model(_original)
        except IndexError:
            _rows.append({"file": _path.name, "status": "no Model: table"})
            continue

        _worst = 0.0
        _worst_term = ""
        for _term, _stats in _orig.items():
            for _stat in ["coeff", "se", "t", "p"]:
                _a, _b = _new.get(_term, {}).get(_stat, np.nan), _stats.get(_stat, np.nan)
                _rel_diff = abs(_a - _b) / max(abs(_b), 1e-12)
                if not np.isfinite(_rel_diff) or _rel_diff > _worst:
                    _worst, _worst_term = (_rel_diff if np.isfinite(_rel_diff) else np.inf), f"{_term} {_stat}"

        # PROCESS prints 4 decimals, so allow rounding error
        _match = _n_new == _n_orig and set(_new) == set(_orig) and _worst < 1e-3
        _rows.append({
            "file": _path.name,
            "status": "match" if _match else "differs",
            "n_new": _n_new,
            "n_original": _n_orig,
            "largest_relative_diff": _worst,
            "at": _worst_term,
        })

    process_comparison = pd.DataFrame(_rows)
    return (process_comparison,)


@app.cell
def _(bu, mo, process_comparison):
    _counts = process_comparison["status"].value_counts().to_dict()
    mo.vstack([
        mo.md(" · ".join(f"**{status}**: {n}" for status, n in _counts.items())),
        bu.show_table(mo, process_comparison),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Figures
    """)
    return


@app.cell
def _(mo, new_files, new_root, original_root):
    _figures = [p for p in new_files("*.png")]
    _options = {str(p.relative_to(new_root)): p for p in _figures}

    figure_picker = mo.ui.dropdown(options=list(_options), value=next(iter(_options), None), label="Figure")
    figure_paths = _options
    figure_picker
    return figure_paths, figure_picker


@app.cell
def _(figure_paths, figure_picker, mo, original_root):
    mo.stop(figure_picker.value is None)

    _new = figure_paths[figure_picker.value]
    _original = original_root / figure_picker.value

    mo.hstack(
        [
            mo.vstack([mo.md("**New**"), mo.image(src=_new, width="100%")]),
            mo.vstack([
                mo.md("**Original**"),
                mo.image(src=_original, width="100%") if _original.exists() else mo.md("_not in original folder_"),
            ]),
        ],
        widths="equal",
    )
    return


if __name__ == "__main__":
    app.run()
