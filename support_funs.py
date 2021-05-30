# SCRIPT FOR FUNCTION WRAPPERS


# import shutil
# import requests

# # For custom packages
# pip install yahoofinance
# pip install git+https://github.com/ianepreston/stats_can


import os
import numpy as np
import pandas as pd
from colorspace.colorlib import HCL

# pip install yahoofinancials
from yahoofinancials import YahooFinancials
# from yahoofinance import BalanceSheet
# import yfinance as yf


def gg_color_hue(n):
    hues = np.linspace(15, 375, num=n + 1)[:n]
    hcl = []
    for h in hues:
        hcl.append(HCL(H=h, L=65, C=100).colors()[0])
    return hcl


def gg_save(fn,fold,gg,width,height):
    path = os.path.join(fold, fn)
    if os.path.exists(path):
        os.remove(path)
    gg.save(path, width=width, height=height)

# Function to remove files if they exist
def rm_if_exists(path):
  if os.path.exists(path):
    print('Removing file: %s' % path)
    os.remove(path)

def ym2date(x):
  assert x.columns.isin(['year','month']).sum() == 2
  return x.assign(date = lambda w: pd.to_datetime(w.year.astype(str)+'-'+w.month.astype(str)))


# ---- Index come column of df (cn_val) to smallest point (cn_idx) --- #
def idx_first(df,cn_gg, cn_idx, cn_val):
  # cn_gg=['city']; cn_idx='date'; cn_val='idx'
  if isinstance(cn_gg, str):
    cn_gg = [cn_gg]
  assert isinstance(cn_idx, str)
  assert isinstance(cn_val, str)
  df = df.copy()
  cn_val_min = cn_val + '_mi'
  idx_min = df.groupby(cn_gg).apply(lambda x: x[cn_idx].idxmin())
  idx_min = idx_min.reset_index().rename(columns={0:'idx'})
  val_min = df.loc[idx_min.idx,cn_gg + [cn_val]]
  val_min.rename(columns={cn_val:cn_val_min}, inplace=True)
  df = df.merge(val_min,'left',cn_gg)
  df = df.assign(val_idx = lambda x: x[cn_val]/x[cn_val_min]*100).drop(columns=[cn_val_min])
  df = df.drop(columns=cn_val).rename(columns={'val_idx':cn_val})
  return df

# ---- Get the stock price history --- #
# ticker, d1, d2, dividend, balance = 'NWH-UN.TO', '2020-01-01', '2021-06-01', True, True
def get_YahooFinancials(ticker, d1, d2, dividend=True, balance=True):
  di = {'formatted_date':'date','open':'price','amount':'dividend'}
  # (1) Get price data
  cn_price = ['formatted_date','open']
  stock = YahooFinancials(ticker)
  price = stock.get_historical_price_data(d1, d2, 'monthly')[ticker]
  price = pd.DataFrame(price['prices'])[cn_price]
  price = price.rename(columns=di).assign(ticker=ticker)
  price.date = pd.to_datetime(price.date)
  price = add_date_int(price)#.drop(columns=['day','date'])
  price = price.query('day == 1').reset_index(None,True).drop(columns='day')
  assert not (price.year + price.month/100).duplicated().any()
  assert price.price.notnull().all()
  # (2) Get dividend data
  if dividend:
    cn_dividend = ['formatted_date','amount']
    dividend = stock.get_daily_dividend_data(d1, d2)[ticker]
    dividend = pd.DataFrame(dividend)[cn_dividend]
    dividend.rename(columns=di,inplace=True)
    dividend.date = pd.to_datetime(dividend.date)
    dividend = add_date_int(dividend)
    dividend = dividend.groupby(['year','month']).dividend.sum().reset_index()
    price = price.merge(dividend,'left',['year','month']) # merge
  return price

# COMPARE YAHOOFINANCIALS TO YFINANCE
# ticker = 'PRV-UN.TO'
def get_yfinance():
  1



# ---- Add YMD to DataFrame --- #
def add_date_int(df):
  df2 = df.assign(year=lambda x: x.date.dt.strftime('%Y').astype(int),
                  month=lambda x: x.date.dt.strftime('%m').astype(int),
                  day=lambda x: x.date.dt.strftime('%d').astype(int))
  cc = ['year','month','day'] + df.columns.to_list()
  df2 = df2[cc]
  return df2

# ---- Make folder --- #
def makeifnot(path):
    if not os.path.exists(path):
        print('path does not exist: %s\nMaking!' % path)
        os.mkdir(path)
    else:
        print('path already exists: %s' % path)

  # # (3) Get balance sheet
  # # if balance:
  # # stock.get_financial_stmts(frequency='annual',statement_type='balance')
  # msft = yf.Ticker('MFST')
  # msft.financials
  # bsheet = BalanceSheet(ticker).to_dfs()
  # tmp_lia = bs['Liabilities'].reset_index()
  # tmp_lia = tmp_lia[tmp_lia.Item == 'Total Liabilities'].melt('Item',None,'date','liability').drop(columns=['Item'])
  # tmp_asset = bs['Assets'].reset_index()
  # tmp_asset = tmp_asset[tmp_asset.Item == 'Total Assets'].melt('Item',None,'date','asset').drop(columns=['Item'])
  # tmp_balance = tmp_asset.merge(tmp_lia)
  # tmp4 = tmp_balance.assign(year=lambda x: pd.to_datetime(x.date).dt.strftime('%Y').astype(int)).drop(columns=['date'])

