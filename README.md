# Just preview on screen, no save
python scada_merge.py --preview

# Preview on screen AND save truncated file
python scada_merge.py --preview --save-preview

# Custom row count, preview + save
python scada_merge.py --preview --preview-rows 500 --save-preview

# Full merge and save (unchanged)
python scada_merge.py
