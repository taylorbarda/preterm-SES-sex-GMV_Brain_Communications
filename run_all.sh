#!/usr/bin/env bash
# Runs the whole pipeline in order and builds a static HTML site of the results.
#
#   ./run_all.sh          fake data  -> outputs/synthetic/ (+ outputs/synthetic/site/index.html)
#   ./run_all.sh real     real data  -> outputs/real/      (+ outputs/real/site/index.html)
#
# The original folder ../mod_analysis_final_bc is only ever read, never written.
set -euo pipefail
cd "$(dirname "$0")"

export BC_DATA_MODE="${1:-synthetic}"
export MPLBACKEND=Agg

PY=.venv/bin/python
MARIMO=.venv/bin/marimo
SITE="outputs/$BC_DATA_MODE/site"

if [[ "$BC_DATA_MODE" == "synthetic" && ! -f synthetic_data/aseg_volume.csv ]]; then
    "$PY" make_synthetic_data.py
fi

NOTEBOOKS=(0[1-6]_*.py)
if [[ "$BC_DATA_MODE" == "real" ]]; then
    NOTEBOOKS+=(07_compare_with_original.py)
fi

mkdir -p "$SITE"

for nb in "${NOTEBOOKS[@]}"; do
    echo "▶ $nb"
    "$PY" "$nb"                                              # stops here if any cell fails
    "$MARIMO" export html "$nb" -o "$SITE/${nb%.py}.html" --force >/dev/null
done

{
    echo "<!doctype html><meta charset=utf-8><title>Brain Communications analysis ($BC_DATA_MODE)</title>"
    echo "<body style='font-family:system-ui;max-width:640px;margin:40px auto;line-height:1.6'>"
    echo "<h1>Brain Communications analysis</h1><p>Data mode: <b>$BC_DATA_MODE</b> · built $(date '+%Y-%m-%d %H:%M')</p><ol>"
    for nb in "${NOTEBOOKS[@]}"; do
        echo "<li><a href='${nb%.py}.html'>${nb%.py}</a></li>"
    done
    echo "</ol></body>"
} > "$SITE/index.html"

echo "Done. Open $SITE/index.html"
