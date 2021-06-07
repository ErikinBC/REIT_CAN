# Load libraries
import os
import pickle
import pandas as pd
import numpy as np

from funs_support import ym2date

# Load the data
dir_base = os.getcwd()

with open('data_for_plots.pickle', 'rb') as handle:
    di_storage = pickle.load(handle)

df_tera = di_storage['teranet']
tera_w = ym2date(di_storage['tera_w'])
df_crea = di_storage['crea']
df_cpi_tsx = di_storage['cpi_tsx']
df_lf = di_storage['lf']
df_mort_tera = di_storage['mort']
df_reit = di_storage['reit']

# QUADRANT CALCULATIONS:
# HOW MANY TIMES WOULD YOU MAKE MONEY WHEN IT FELL
# HOW MANY TIMES WILL YOU LOSE
# NOW DO THIS FOR A MAGNITUDE

