import pandas as pd
from pathlib import Path
import glob
import chardet
import csv
import re
import argparse
import tempfile
import os

# ────────────────────────────────────────────────
#  CLI Arguments
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Merge multiple sensor CSV files on a shared Time column."
)
parser.add_argument("--preview",       action="store_true",
    help="Preview first N rows of merged result without saving.")
parser.add_argument("--preview-rows",  type=int, default=10000, metavar="N",
    help="Number of rows to preview (default: 10000).")
parser.add_argument("--save-preview",  action="store_true",
    help="Save the preview-truncated result (requires --preview).")
parser.add_argument("--folder",        type=str, default=r"./",
    help="Folder path containing CSV files.")
parser.add_argument("--pattern",       type=str, default="*.csv",
    help="File glob pattern (default: *.csv).")
parser.add_argument("--output",        type=str, default="merged_sensors_2026.csv",
    help="Output filename.")
parser.add_argument("--chunk-size",    type=int, default=100000, metavar="N",
    help="Rows per chunk during merge (default: 100000). Lower = less RAM.")
args = parser.parse_args()

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
name_from           = "prefix"

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
    "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be",
    "cp1252", "cp1250", "iso-8859-1", "iso-8859-2",
    "latin1",
]

def read_csv_auto(filepath, **kwargs):
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
        f"Could not parse '{filename}' into 2+ columns with any known encoding/delimiter."
    )

# ────────────────────────────────────────────────
#  Chunked merge: streams through temp CSVs
#  instead of holding everything in RAM
# ────────────────────────────────────────────────
def chunked_merge(base_df, new_df, chunk_size, output_path, write_header):
    """
    Merges base_df with new_df in chunks, writing rows directly to output_path.
    Returns the number of rows written.
    """
    # Sort both frames by Time once
    base_df = base_df.sort_values("Time").reset_index(drop=True)
    new_df  = new_df.sort_values("Time").reset_index(drop=True)

    rows_written = 0
    mode = "w" if write_header else "a"

    for start in range(0, len(base_df), chunk_size):
        chunk = base_df.iloc[start : start + chunk_size]

        # Get time window for this chunk
        t_min = chunk["Time"].min()
        t_max = chunk["Time"].max()

        # Slice only the relevant rows from new_df
        mask        = (new_df["Time"] >= t_min) & (new_df["Time"] <= t_max)
        new_slice   = new_df[mask]

        merged_chunk = pd.merge(chunk, new_slice, on="Time", how="outer")
        merged_chunk = merged_chunk.sort_values("Time")

        merged_chunk.to_csv(
            output_path,
            index=False,
            mode=mode,
            header=(rows_written == 0),
            date_format="%Y-%m-%d %H:%M:%S"
        )
        rows_written += len(merged_chunk)
        mode = "a"  # append after first write

    # Flush any rows in new_df beyond the time range of base_df
    t_base_max  = base_df["Time"].max()
    tail_mask   = new_df["Time"] > t_base_max
    tail        = new_df[tail_mask]
    if not tail.empty:
        tail.to_csv(
            output_path,
            index=False,
            mode="a",
            header=False,
            date_format="%Y-%m-%d %H:%M:%S"
        )
        rows_written += len(tail)

    return rows_written

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
    mode_label = "MERGE & SAVE MODE (chunked)"

print(f"{'─'*55}")
print(f"  {mode_label}")
print(f"  Folder     : {Path(folder_path).resolve()}")
print(f"  Pattern    : {file_pattern}")
print(f"  Chunk size : {args.chunk_size:,} rows")
if args.preview:
    print(f"  Rows       : first {args.preview_rows:,} per file (optimised read)")
if not args.preview or args.save_preview:
    print(f"  Output     : {output_file}")
print(f"{'─'*55}\n")
print(f"Found {len(files)} file(s)\n")

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

    df = df.rename(columns={time_col: "Time", value_col: value_new_name})

    # Downcast float64 → float32 to halve memory per column
    for col in df.select_dtypes(include="float64").columns:
        df[col] = df[col].astype("float32")

    dfs.append(df)

if skipped:
    print(f"\n⚠ Skipped {len(skipped)} file(s):")
    for s in skipped:
        print(f"   - {Path(s).name}")

if not dfs:
    print("\nNo files were successfully loaded. Exiting.")
    exit()

# ────────────────────────────────────────────────
#  Merge — chunked to avoid RAM explosion
# ────────────────────────────────────────────────
if args.preview and not args.save_preview:
    # In screen-only preview: do a normal in-memory merge (rows already capped)
    print(f"\nMerging {len(dfs)} dataframe(s) in memory (preview, rows capped)...")
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Time", how="outer", sort=True)
    merged = merged.sort_values("Time").reset_index(drop=True)
    save_path = None

else:
    # Full merge or save-preview: stream through temp file then rename
    tmp_path = output_file + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"\nMerging {len(dfs)} dataframe(s) in chunks of {args.chunk_size:,}...")

    # Start with first dataframe, write to temp file in chunks
    base = dfs[0]
    for i, df in enumerate(dfs[1:], start=2):
        print(f"  Merging file {i}/{len(dfs)}...")

        # Use a fresh temp file each round to avoid re-reading huge files
        round_tmp = output_file + f".round{i}.tmp"
        chunked_merge(base, df, args.chunk_size, round_tmp, write_header=True)

        # Read the round result back as the new base
        # Use float32 to keep memory low
        base = pd.read_csv(
            round_tmp,
            parse_dates=["Time"],
            dtype={c: "float32" for c in renamed_value_cols if c != "Time"}
        )
        os.remove(round_tmp)
        print(f"    ✓ Running shape after merge: {base.shape}")

    merged   = base.sort_values("Time").reset_index(drop=True)
    save_path = output_file

    if args.preview and args.save_preview:
        save_path = Path(output_file).stem + "_preview" + Path(output_file).suffix

# ────────────────────────────────────────────────
#  Preview or Save
# ────────────────────────────────────────────────
print(f"\nFinal shape : {merged.shape}")
print(f"Time range  : {merged['Time'].min()}  →  {merged['Time'].max()}")
print(f"Sensors     : {list(merged.columns[1:])}")

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
        merged.head(n).to_csv(save_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
        print(f"  Preview saved to : {save_path}")
        print(f"  Rows written     : {n:,}")
    else:
        print(f"  Preview complete. Add --save-preview to save, or remove --preview to save full merge.")
    print(f"{'─'*55}")

else:
    merged.to_csv(save_path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(f"\nSaved to: {save_path}")
