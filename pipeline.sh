#!/bin/bash

echo "Script 1: Real-time REITs"
python get_REIT.py
# output: 

echo "Script 2: Process data"
python process_data.py
# output: date_for_plots.pickle

echo "Script 3: Statistical associations"
python run_stats.py
# output: 

echo "Script 4: Make figures"
python make_figs.py
# output: ~/figures/*.png

echo "END OF PIPELINE.SH"

