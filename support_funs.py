"""
SCRIPT FOR FUNCTION WRAPPERS
"""

import os
import numpy as np
import pandas as pd

def add_date_int(df):
  df2 = df.assign(year=lambda x: x.date.dt.strftime('%Y').astype(int),
                  month=lambda x: x.date.dt.strftime('%m').astype(int),
                  day=lambda x: x.date.dt.strftime('%d').astype(int))
  cc = ['year','month','day'] + df.columns.to_list()
  df2 = df2[cc]
  return df2

def makeifnot(path):
    if not os.path.exists(path):
        print('path does not exist: %s\nMaking!' % path)
        os.mkdir(path)
    else:
        print('path already exists: %s' % path)
#


