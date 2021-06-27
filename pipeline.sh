#!/bin/bash

conda activate reit

echo "Script 1: Process data"
python 1_process_data.py
# output: di_storage.pickle

echo "Script 2: Make figures"
python 2_make_figs.py
# output:   ~/figures/{gg_shiller_mm, gg_shiller_idx, gg_drate_other, gg_tera_other, gg_tera_other_mm, gg_tera_quad, gg_tera_w, gg_tera_crea_lvl, gg_tera_crea_pct, gg_rho_hpi, gg_crea_tt, gg_hpi_vs_stats, gg_lf_share, gg_mort_tera, gg_arate_dividend, gg_perf_div, gg_err_div, gg_best_err, gg_reit_idx, gg_reit_idx_bs}.png
#           ~/reit_bs.csv

echo "Script 3: Shorting statrategy"
python 3_run_short.py
# output: ~/figures/{gg_hpi_sidx, gg_quadrant_{metric}, gg_short_{city}, gg_rank_{metric}, gg_comp_short, gg_comp_metric}.png

echo "END OF PIPELINE.SH"

