import pandas as pd
from pathlib import Path
import glob
import chardet
import csv
import re
import argparse
import os
import heapq
import time

# ────────────────────────────────────────────────
#  CLI Arguments
# ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Merge multiple sensor CSV files on a shared Time column."
)
parser.add_argument("--preview",      action="store_true",
    help="Preview first N rows of merged result without saving.")
parser.add_argument("--preview-rows", type=int, default=10000, metavar="N",
    help="Number of rows to preview (default: 10000).")
parser.add_argument("--save-preview", action="store_true",
    help="Save the preview-truncated result (requires --preview).")
parser.add_argument("--folder",       type=str, default=r"./",
    help="Folder path containing CSV files.")
parser.add_argument("--pattern",      type=str, default="*.csv",
    help="File glob pattern (default: *.csv).")
parser.add_argument("--output",       type=str, default="merged_sensors_2026.csv",
    help="Output filename.")
parser.add_argument("--chunk-size",   type=int, default=100000, metavar="N",
    help="Rows per chunk during incremental write (default: 100000).")
args = parser.parse_args()

if args.save_preview and not args.preview:
    parser.error("--save-preview requires --preview")

# ────────────────────────────────────────────────
#  Elapsed time helper
# ────────────────────────────────────────────────
_t0 = time.perf_counter()   # script start

def elapsed(since=None):
    """Return formatted elapsed time string from `since` (default: script start)."""
    secs = time.perf_counter() - (since if since is not None else _t0)
    if secs < 60:
        return f"{secs:.1f}s"
    m, s = divmod(int(secs), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"

def ts():
    """Inline elapsed tag for log lines, e.g. [+12.3s]"""
    return f"[+{elapsed()}]"

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
#  Helper: robust encoding + delimiter CSV read
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
#  Helper: sorted temp CSV writer
# ────────────────────────────────────────────────
def write_sorted_temp(df, path):
    df = df.sort_values("Time").reset_index(drop=True)
    df.to_csv(path, index=False, date_format="%Y-%m-%d %H:%M:%S")
    return path

# ────────────────────────────────────────────────
#  Core: incremental k-way merge
# ────────────────────────────────────────────────
def incremental_kway_merge(dfs, all_columns, output_path, chunk_size, preview_rows=None):
    tmp_dir   = Path(output_path).parent
    tmp_files = []

    # Step 1: write each df sorted to its own temp file
    t_sort = time.perf_counter()
    print(f"\n  {ts()} Writing sorted temp files...")
    for i, df in enumerate(dfs):
        tmp_path = str(tmp_dir / f"_merge_tmp_{i}.csv")
        write_sorted_temp(df, tmp_path)
        tmp_files.append(tmp_path)
        print(f"    ✓ Temp {i+1}/{len(dfs)}: {len(df):,} rows  [{elapsed(t_sort)}]")

    # Step 2: open chunk iterators per temp file
    iterators = [
        pd.read_csv(p, parse_dates=["Time"], chunksize=chunk_size, dtype="object")
        for p in tmp_files
    ]

    # Step 3: initialise heap with first row of each file
    heap         = []
    chunk_buffer = {}
    exhausted    = set()

    def load_next(file_idx):
        if file_idx in exhausted:
            return False
        buf = chunk_buffer.get(file_idx, [])
        while not buf:
            try:
                chunk = next(iterators[file_idx])
                buf   = chunk.to_dict("records")
            except StopIteration:
                exhausted.add(file_idx)
                chunk_buffer[file_idx] = []
                return False
        row = buf.pop(0)
        chunk_buffer[file_idx] = buf
        heapq.heappush(heap, (str(row["Time"]), file_idx, row))
        return True

    for i in range(len(dfs)):
        load_next(i)

    # Step 4: stream rows in time order, write in chunks
    t_merge      = time.perf_counter()
    write_buffer = []
    rows_written = 0
    first_write  = True
    limit_reached = False
    current_time = None
    current_row  = {}

    def flush_buffer():
        nonlocal first_write, rows_written
        if not write_buffer:
            return
        out_df = pd.DataFrame(write_buffer, columns=["Time"] + all_columns)
        out_df["Time"] = pd.to_datetime(out_df["Time"])
        for col in all_columns:
            out_df[col] = pd.to_numeric(out_df[col], errors="coerce").astype("float32")
        out_df.sort_values("Time").to_csv(
            output_path,
            index=False,
            mode="w" if first_write else "a",
            header=first_write,
            date_format="%Y-%m-%d %H:%M:%S"
        )
        rows_written += len(out_df)
        first_write   = False
        write_buffer.clear()
        rate = rows_written / max(time.perf_counter() - t_merge, 0.001)
        print(f"    ✓ {rows_written:>10,} rows written  "
              f"[+{elapsed()}]  ({rate:,.0f} rows/s)      ", end="\r")

    while heap and not limit_reached:
        ts_val, file_idx, row = heapq.heappop(heap)

        if ts_val != current_time:
            if current_time is not None:
                write_buffer.append(current_row)
                if len(write_buffer) >= chunk_size:
                    flush_buffer()
                    if preview_rows and rows_written >= preview_rows:
                        limit_reached = True
                        break
            current_time = ts_val
            current_row  = {"Time": ts_val}

        for col in all_columns:
            if col in row and row[col] not in (None, "", "nan"):
                current_row[col] = row[col]

        load_next(file_idx)

    if current_time is not None and not limit_reached:
        write_buffer.append(current_row)
    flush_buffer()

    # Step 5: cleanup temp files
    for path in tmp_files:
        try:
            os.remove(path)
        except OSError:
            pass

    print()  # clear \r line
    return rows_written

# ────────────────────────────────────────────────
#  Main script
# ────────────────────────────────────────────────
files = sorted(glob.glob(str(Path(folder_path) / file_pattern)))
if not files:
    print("No files found. Check folder_path and file_pattern.")
    exit()

if args.preview and args.save_preview:
    mode_label = "PREVIEW + SAVE MODE"
elif args.preview:
    mode_label = "PREVIEW MODE (no file saved)"
else:
    mode_label = "MERGE & SAVE MODE (incremental)"

print(f"{'─'*55}")
print(f"  {mode_label}")
print(f"  Folder     : {Path(folder_path).resolve()}")
print(f"  Pattern    : {file_pattern}")
print(f"  Chunk size : {args.chunk_size:,} rows")
if args.preview:
    print(f"  Rows       : first {args.preview_rows:,} per file")
if not args.preview or args.save_preview:
    print(f"  Output     : {output_file}")
print(f"{'─'*55}\n")
print(f"Found {len(files)} file(s)\n")

# ── Read phase ────────────────────────────────────
t_read = time.perf_counter()
print(f"{ts()} Reading files...")

read_kwargs = dict(
    parse_dates=[0],
    date_format="%m/%d/%Y %I:%M:%S %p",
    dayfirst=False,
    cache_dates=True,
)
if args.preview:
    read_kwargs["nrows"] = args.preview_rows

dfs                = []
all_value_cols     = []
renamed_value_cols = set()
skipped            = []

for f in files:
    t_file = time.perf_counter()
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
    all_value_cols.append(value_new_name)

    df = df.rename(columns={time_col: "Time", value_col: value_new_name})
    df = df[["Time", value_new_name]]
    dfs.append(df)
    print(f"  ↳ {len(df):,} rows loaded  [{elapsed(t_file)}]")

print(f"\n{ts()} All files read  (read phase: {elapsed(t_read)})")

if skipped:
    print(f"\n⚠ Skipped {len(skipped)} file(s):")
    for s in skipped:
        print(f"   - {Path(s).name}")

if not dfs:
    print("\nNo files were successfully loaded. Exiting.")
    exit()

# ── Merge phase ───────────────────────────────────
t_merge = time.perf_counter()
print(f"\n{ts()} Starting incremental k-way merge of {len(dfs)} file(s)...")

if args.preview and not args.save_preview:
    print("  (in-memory merge — rows capped by --preview-rows)\n")
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on="Time", how="outer", sort=True)
    merged       = merged.sort_values("Time").reset_index(drop=True)
    save_path    = None
    rows_written = len(merged)
else:
    save_path = (Path(output_file).stem + "_preview" + Path(output_file).suffix
                 if args.save_preview else output_file)
    preview_limit = args.preview_rows if args.save_preview else None
    rows_written  = incremental_kway_merge(
        dfs, all_value_cols, save_path, args.chunk_size, preview_rows=preview_limit
    )
    merged = None

print(f"{ts()} Merge complete  (merge phase: {elapsed(t_merge)})")

# ── Output phase ──────────────────────────────────
t_out = time.perf_counter()
print(f"\n{'─'*55}")

if args.preview:
    display_df = (
        merged.head(args.preview_rows)
        if merged is not None
        else pd.read_csv(save_path, parse_dates=["Time"], nrows=args.preview_rows)
    )
    n = len(display_df)
    print(f"  PREVIEW — {n:,} rows")
    print(f"{'─'*55}\n")
    with pd.option_context(
        "display.max_rows",     n,
        "display.max_columns",  None,
        "display.width",        None,
        "display.float_format", "{:.4f}".format
    ):
        print(display_df.to_string(index=True))
    print(f"\n{'─'*55}")
    if args.save_preview:
        print(f"  Preview saved to : {save_path}")
        print(f"  Rows written     : {rows_written:,}")
    else:
        print(f"  Preview complete. Add --save-preview to save, or remove --preview to save full merge.")
else:
    print(f"  Saved to         : {save_path}")
    print(f"  Rows written     : {rows_written:,}")

# ── Timing summary ────────────────────────────────
print(f"{'─'*55}")
print(f"  Timing summary")
print(f"{'─'*55}")
print(f"  Read phase   : {elapsed(t_read)}")
print(f"  Merge phase  : {elapsed(t_merge)}")
if not args.preview or args.save_preview:
    print(f"  Write phase  : {elapsed(t_out)}")
print(f"  Total        : {elapsed()}")
print(f"{'─'*55}")
