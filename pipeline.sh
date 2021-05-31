#!/bin/bash

echo "Script 1: Process data"
python process_data.py
# output: date_for_plots.pickle

echo "Script 2: Statistical associations"
#python run_stats.py
# output: 

echo "Script 3: Make figures"
python make_figs.py
# output: ~/figures/*.png

echo "END OF PIPELINE.SH"

