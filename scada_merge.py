import pandas as pd
from pathlib import Path
import glob
import chardet
import csv
import re
import argparse

# ────────────────────────────────────────────────
#  CLI Arguments
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Merge multiple sensor CSV files on a shared Time column."
)
parser.add_argument(
    "--preview",
    action="store_true",
    help="Preview first N rows of merged result without saving."
)
parser.add_argument(
    "--preview-rows",
    type=int,
    default=10000,
    metavar="N",
    help="Number of rows to preview (default: 10000). Only used with --preview."
)
parser.add_argument(
    "--save-preview",
    action="store_true",
    help="Save the preview-truncated result to <output>_preview.csv (requires --preview)."
)
parser.add_argument(
    "--folder",
    type=str,
    default=r"./",
    help="Folder path containing CSV files (default: current directory)."
)
parser.add_argument(
    "--pattern",
    type=str,
    default="*.csv",
    help="File glob pattern (default: *.csv)."
)
parser.add_argument(
    "--output",
    type=str,
    default="merged_sensors_2026.csv",
    help="Output filename (default: merged_sensors_2026.csv)."
)
args = parser.parse_args()

# ── Validate: --save-preview only makes sense with --preview ──
if args.save_preview and not args.preview:
    parser.error("--save-preview requires --preview")

# ────────────────────────────────────────────────
#  Settings
# ────────────────────────────────────────────────
folder_path         = args.folder
file_pattern        = args.pattern
output_file         = args.output
time_column_suffix  = " Time"
value_column_suffix = " ValueY"
name_from           = "prefix"      # "prefix" or "filename"

# ────────────────────────────────────────────────
#  Helper: detect delimiter
# ────────────────────────────────────────────────
def detect_delimiter(filepath, encoding):
    try:
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","

# ────────────────────────────────────────────────
#  Helper: read CSV with robust encoding + delimiter
# ────────────────────────────────────────────────
FALLBACK_ENCODINGS = [
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1252",
    "cp1250",
    "iso-8859-1",
    "iso-8859-2",
    "latin1",       # never fails – always last resort
]

def read_csv_auto(filepath, **kwargs):
    """
    Auto-detects encoding and delimiter, then reads the CSV.
    Any kwargs (e.g. nrows, parse_dates) are forwarded to pd.read_csv.
    """
    filename = Path(filepath).name

    with open(filepath, "rb") as f:
        raw = f.read()
    detected    = chardet.detect(raw)
    chardet_enc = detected.get("encoding")
    confidence  = detected.get("confidence", 0)

    encodings_to_try = []
    if chardet_enc:
        encodings_to_try.append(chardet_enc)
    for enc in FALLBACK_ENCODINGS:
        if enc.lower() not in [e.lower() for e in encodings_to_try]:
            encodings_to_try.append(enc)

    for enc in encodings_to_try:
        try:
            sep = detect_delimiter(filepath, enc)
            df  = pd.read_csv(filepath, encoding=enc, sep=sep, **kwargs)

            if df.shape[1] < 2:
                print(f"  ✗ [{enc}] only {df.shape[1]} col(s) with sep='{sep}' → retrying")
                continue

            nrows_info = f" | nrows={kwargs['nrows']}" if "nrows" in kwargs else ""
            print(f"  ✓ [{enc}] sep='{sep}' | {df.shape[1]} cols{nrows_info} "
                  f"(chardet: {chardet_enc} @ {confidence:.0%}) → {filename}")
            return df

        except (UnicodeDecodeError, LookupError):
            continue

    print(f"\n  ✗ FAILED to parse '{filename}'. Raw byte preview:")
    with open(filepath, "rb") as f:
        print(f.read(200))
    raise ValueError(
        f"Could not parse '{filename}' into 2+ columns with any known encoding/delimiter.\n"
        f"Check the file manually — it may be empty, binary, or have an unusual format."
    )

# ────────────────────────────────────────────────
#  Main script
# ────────────────────────────────────────────────
files = sorted(glob.glob(str(Path(folder_path) / file_pattern)))
if not files:
    print("No files found. Check folder_path and file_pattern.")
    exit()

if args.preview and args.save_preview:
    mode_label = "PREVIEW + SAVE MODE (truncated file will be saved)"
elif args.preview:
    mode_label = "PREVIEW MODE (no file will be saved)"
else:
    mode_label = "MERGE & SAVE MODE"

print(f"{'─'*55}")
print(f"  {mode_label}")
print(f"  Folder  : {Path(folder_path).resolve()}")
print(f"  Pattern : {file_pattern}")
if args.preview:
    print(f"  Rows    : first {args.preview_rows:,} per file (optimised read)")
if not args.preview or args.save_preview:
    print(f"  Output  : {output_file}")
print(f"{'─'*55}\n")
print(f"Found {len(files)} file(s)\n")

# ── In preview mode, cap each file read to preview_rows ──────
read_kwargs = dict(
    parse_dates=[0],
    date_format="%m/%d/%Y %I:%M:%S %p",
    dayfirst=False,
    cache_dates=True,
)
if args.preview:
    read_kwargs["nrows"] = args.preview_rows

dfs                = []
renamed_value_cols = set()
skipped            = []

for f in files:
    print(f"Reading: {Path(f).name}")
    try:
        df = read_csv_auto(f, **read_kwargs)
    except ValueError as e:
        print(f"  ⚠ Skipping: {e}\n")
        skipped.append(f)
        continue

    time_col  = df.columns[0]
    value_col = df.columns[1]

    if name_from == "prefix":
        sensor = time_col.replace(time_column_suffix, "").strip()
    else:
        sensor = Path(f).stem.strip()

    # Generic "Trend X" header → fall back to filename
    if re.match(r"^Trend\s*\d+$", sensor, re.IGNORECASE):
        sensor = Path(f).stem.strip()
        print(f"  ↳ Generic header detected → using filename: '{sensor}'")

    value_new_name = sensor.replace(" ", "_").strip()

    if value_new_name in renamed_value_cols:
        print(f"  ⚠ Duplicate sensor name '{value_new_name}' → appending suffix")
        i = 2
        while f"{value_new_name}_{i}" in renamed_value_cols:
            i += 1
        value_new_name = f"{value_new_name}_{i}"
    renamed_value_cols.add(value_new_name)

    df = df.rename(columns={
        time_col:  "Time",
        value_col: value_new_name
    })
    dfs.append(df)

if skipped:
    print(f"\n⚠ Skipped {len(skipped)} file(s):")
    for s in skipped:
        print(f"   - {Path(s).name}")

if not dfs:
    print("\nNo files were successfully loaded. Exiting.")
    exit()

# ────────────────────────────────────────────────
#  Merge
# ────────────────────────────────────────────────
print(f"\nMerging {len(dfs)} dataframe(s)...")
merged = dfs[0]
for df in dfs[1:]:
    merged = pd.merge(
        merged,
        df,
        on="Time",
        how="outer",
        sort=True
    )

merged = merged.sort_values("Time").reset_index(drop=True)

# Optional: convert to Singapore timezone
# merged["Time"] = merged["Time"].dt.tz_localize("UTC").dt.tz_convert("Asia/Singapore")

print(f"\nFinal shape : {merged.shape}")
print(f"Time range  : {merged['Time'].min()}  →  {merged['Time'].max()}")
print(f"Sensors     : {list(merged.columns[1:])}")

# ────────────────────────────────────────────────
#  Preview or Save
# ────────────────────────────────────────────────
if args.preview:
    n = min(args.preview_rows, len(merged))
    print(f"\n{'─'*55}")
    print(f"  PREVIEW — first {n:,} of {len(merged):,} rows")
    print(f"{'─'*55}\n")
    with pd.option_context(
        "display.max_rows",     n,
        "display.max_columns",  None,
        "display.width",        None,
        "display.float_format", "{:.4f}".format
    ):
        print(merged.head(n).to_string(index=True))
    print(f"\n{'─'*55}")

    if args.save_preview:
        preview_file = Path(output_file).stem + "_preview" + Path(output_file).suffix
        merged.head(n).to_csv(preview_file, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"  Preview saved to : {preview_file}")
        print(f"  Rows written     : {n:,}")
    else:
        print(f"  Preview complete. Add --save-preview to save, or remove --preview to save full merge.")

    print(f"{'─'*55}")
else:
    merged.to_csv(output_file, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\nSaved to: {output_file}")
