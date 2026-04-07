# wincc_csv_merge
Merge Siemens WinCC csv files

# Normal merge and save
python scada_merge.py

# Preview first 10,000 rows (no file written)
python scada_merge.py --preview

# Preview a custom number of rows
python scada_merge.py --preview --preview-rows 500

# Override folder, pattern, output on the fly
python scada_merge.py --folder "C:\data\scada" --pattern "2026_*.csv" --output merged_jan.csv

# Preview with custom folder
python scada_merge.py --preview --folder "C:\data\scada" --pattern "5_HF_*.csv"
