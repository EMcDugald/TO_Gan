#!/usr/bin/env bash
set -euo pipefail

PYTHON="$HOME/.conda/envs/to_gan/bin/python"
CSV_PATH="$HOME/TO_Gan/TO_3D_data_scratch/data/summary.csv"
SCALAR_COLUMN="deformation_p99"

PERCENTILE="${1:?Usage: $0 <percentile-0-to-100> [csv_path] [scalar_column]}"
CSV_PATH="${2:-$CSV_PATH}"
SCALAR_COLUMN="${3:-$SCALAR_COLUMN}"

"$PYTHON" - "$CSV_PATH" "$SCALAR_COLUMN" "$PERCENTILE" <<'PY'
import csv
import sys
import numpy as np

csv_path, scalar_col, percentile_str = sys.argv[1:]
percentile = float(percentile_str)

if not 0.0 <= percentile <= 100.0:
    raise ValueError(f"Percentile must be in [0, 100], got {percentile}")

values_by_part = {}

with open(csv_path, newline="") as f:
    content_lines = (line for line in f if not line.lstrip().startswith("#"))
    reader = csv.DictReader(content_lines)

    if reader.fieldnames is None or scalar_col not in reader.fieldnames:
        raise ValueError(
            f"Column {scalar_col!r} not found in {csv_path}. "
            f"Available columns: {reader.fieldnames}"
        )

    for row in reader:
        raw = (row.get(scalar_col) or "").strip()
        if raw == "" or raw.lower() == "nan":
            continue

        try:
            value = float(raw)
            part_idx = int(float(row["part"]))
        except (KeyError, TypeError, ValueError):
            continue

        if np.isfinite(value):
            values_by_part[part_idx] = value

if not values_by_part:
    raise ValueError(
        f"No finite values found for column {scalar_col!r} in {csv_path}"
    )

values = np.array(
    [values_by_part[idx] for idx in sorted(values_by_part)],
    dtype=np.float64,
)

threshold = float(np.percentile(np.sort(values), percentile))

print(
    f"MANUF_THRESHOLD_RAW={threshold:.17g} "
    f"percentile={percentile:.1f} "
    f"column={scalar_col} "
    f"n_labeled={len(values)}"
)
PY