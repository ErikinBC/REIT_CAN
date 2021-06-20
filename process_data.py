"""
SCRIPT TO GENERATE THE HOUSE PRICE INDEX AND STATSCAN DEMOGRAPHIC DATA
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

import pandas_datareader as pdr
from stats_can import StatsCan
sc = StatsCan()
import pickle

#get_YahooFinancials, get_yfinance
from funs_support import add_date_int, get_price_dividend, idx_first, rm_if_exists

dir_base = os.getcwd()

# Begin analysis at 2005 for CREA (lines up with StatsCan)
dstart = '2005-01-01'
dstart_dt = pd.to_datetime(dstart)
ystart = int(pd.to_datetime(dstart).strftime('%Y'))
dnow = datetime.now().strftime('%Y-%m-%d')
print('Current date: %s' % dnow)

###############################
### --- (1) TERANET HPI --- ###

cn_ym = ['year','month']

# Download Teranet data (assume Linux/WSL)
fn_tera = 'House_Price_Index.csv'
url_tera = 'https://housepriceindex.ca/_data/' + fn_tera
rm_if_exists(fn_tera)
os.system('wget ' + url_tera + ' --no-check-certificate')

df_tera = pd.read_csv('House_Price_Index.csv',header=[0,1])
idx = pd.IndexSlice
df_tera.rename(columns={'Unnamed: 0_level_1':'Index'},level=1,inplace=True)
# Separate sales and index
tmp1 = df_tera.loc[:,idx[:,'Sales Pair Count']]
tmp1.columns = tmp1.columns.droplevel(1)
tmp2 = df_tera.loc[:,idx[:,'Index']]
tmp2.columns = tmp2.columns.droplevel(1)
tmp2.rename(columns={'Transaction Date':'date','c11':'cad_canada'},inplace=True)
tmp1.columns = tmp2.columns[1:]
tmp1.insert(0,'date',tmp2.date)
df_tera = tmp1.melt('date',None,'city','sales').merge(tmp2.melt('date',None,'city','idx'))
df_tera.date = pd.to_datetime(df_tera.date,format='%b-%Y')
df_tera = df_tera.query('date >= @dstart_dt').reset_index(None,True)
df_tera.sales = df_tera.sales.astype(int)
df_tera = add_date_int(df_tera).drop(columns=['day'])
# Index at 2001-01
df_tera = idx_first(df=df_tera, cn_gg='city', cn_idx='date', cn_val='idx')
# Clean up city
df_tera.city = df_tera.city.str.split('\\_',1,True).iloc[:,1].str.replace('\\_',' ').str.capitalize()
df_tera.city = df_tera.city.str.split('\\s',1,True)[0]
df_tera.insert(3,'tt','aggregate')
# Calculate monthly sales
sales_city = df_tera.query('city != "Canada"').groupby(cn_ym).sales.sum()
sales_city = sales_city.reset_index().rename(columns={'sales':'cities'})
sales_city = sales_city.merge(df_tera.query('city == "Canada"')[cn_ym+['sales']])
assert np.all(sales_city.cities == sales_city.sales)
sales_city.drop(columns='sales',inplace=True)

# Does index equal weighted change?
df_tera_w = df_tera.assign(mm=lambda x: x.idx/x.groupby('city').idx.shift(1)-1)
df_tera_w = df_tera_w[['year','month','city','sales','mm']]
mm_cad = df_tera_w.query('city == "Canada"')[cn_ym+['mm']]#.dropna()
mm_cad.rename(columns={'mm':'mm_cad'},inplace=True)
df_tera_w = df_tera_w.query('city != "Canada"').merge(sales_city)#.dropna()
df_tera_w = df_tera_w.reset_index(None,True).assign(share=lambda x: x.sales/x.cities)
df_tera_w.drop(columns=['sales','cities'],inplace=True)
df_tera_w = df_tera_w.groupby(cn_ym).apply(lambda x: np.sum(x.share*x.mm)).reset_index()
df_tera_w.rename(columns={0:'mm_cities'},inplace=True)
mm_cities = df_tera_w.merge(mm_cad)
# Accumulate growth and compare
mm_cities.loc[0,['mm_cities','mm_cad']] = 0
mm_cities = mm_cities.melt(['year','month'],None,'tt').assign(value=lambda x: x.value+1)
tmp = mm_cities.groupby('tt').apply(lambda x: np.cumproduct(x.value)).reset_index().sort_values('level_1')
mm_cities['idx'] = tmp.value.values

############################
### --- (2) CREA HPI --- ###

# CREA HPI (assume Linux/WSL)
crea_files = ['MLS®-HPI.zip', 'Seasonally Adjusted.xlsx', 'Not Seasonally Adjusted.xlsx']

for file in crea_files:
  rm_if_exists(file)

fn_crea = 'MLS%C2%AE-HPI.zip'
url_crea = 'https://www.crea.ca/wp-content/uploads/2019/06/' + fn_crea
os.system('wget ' + url_crea + ' --no-check-certificate')
os.system('unzip ' + crea_files[0])

# Load in all sheets
cn_crea_from = ['Date','Composite_HPI','Single_Family_HPI','Townhouse_HPI','Apartment_HPI']
cn_crea = ['date','aggregate','sfd','row','apt']
xls_crea = pd.read_excel('Not Seasonally Adjusted.xlsx',sheet_name=None, engine='openpyxl')
holder = []
for city in list(xls_crea):
  if xls_crea[city].columns.isin(cn_crea_from).sum()==len(cn_crea_from):
    print('City: %s' % city)
    holder.append(xls_crea[city][cn_crea_from].assign(city=city))
# Merge and clean up
df_crea = pd.concat(holder).rename(columns=dict(zip(cn_crea_from,cn_crea))).reset_index(None,True)
# Clean up city
df_crea.city = df_crea.city.str.replace('Aggregate','Canada')
df_crea.city = df_crea.city.str.replace('Greater\\_|\\_CMA','',regex=True)
# Get the overlapping cities
hpi_cities = np.intersect1d(df_tera.city.unique(),df_crea.city.unique())
df_crea = df_crea.query('city.isin(@hpi_cities)',engine='python').reset_index(None,True)
df_tera = df_tera.query('city.isin(@hpi_cities)',engine='python').reset_index(None,True)
df_crea = df_crea.melt(['date','city'],None,'tt','idx')

#############################
### --- (3) CPI & TSX --- ###

# (i) All-Items CPI (Statistics Canada)
# Note, may need to delete stats_can.h5 to get up-to-date table
cn_cpi = ['REF_DATE','VALUE','Products and product groups']
di_cpi = dict(zip(cn_cpi, ['date','cpi','products']))
df_cpi = sc.table_to_df('18-10-0006-01')[cn_cpi].rename(columns=di_cpi)
df_cpi.date = pd.to_datetime(df_cpi.date)
df_cpi = df_cpi.query('products=="All-items" & date >= @dstart').reset_index(None,True).drop(columns='products')
df_cpi = df_cpi.assign(cpi = lambda x: x.cpi / x.cpi[0] * 100)

# Get TSX index ticker
ticker_tsx = '^GSPTSE'
df_tsx = get_price_dividend(ticker_tsx,dstart,dnow)
df_tsx = df_tsx[['date','price']].rename(columns={'price':'tsx'})
df_tsx = df_tsx.assign(tsx = lambda x: x.tsx / x.tsx[0] * 100)
df_cpi_tsx = df_cpi.merge(df_tsx,'right','date')

#######################################
### --- (4) STATSCAN POPULATION --- ###

di_cn = {'GEO':'geo','Age group':'age','REF_DATE':'date',
                   'Labour force characteristics':'lf', 'VALUE':'value'}
cn = ['date','geo','value','lf']

di_city = {'Vancouver':'Van/Vic','Victoria':'Van/Vic',
          'Calgary':'Cal/Edm','Edmonton':'Cal/Edm',
          'Montréal':'Montreal', 'Toronto':'GTAH','Hamilton':'GTAH'}

# LOAD CITY-LEVEL
dat_lf_metro = sc.table_to_df('14-10-0096-01')
dat_lf_metro.rename(columns=di_cn,inplace=True)
dat_lf_metro = dat_lf_metro[(dat_lf_metro.Sex=='Both sexes') & (dat_lf_metro.age=='15 years and over') & dat_lf_metro.lf.isin(['Population','Employment'])][cn]
dat_lf_metro['year'] = dat_lf_metro.date.dt.strftime('%Y').astype(int)
# Remove the Quebec/Ontario part of Ottawa
dat_lf_metro = dat_lf_metro[~dat_lf_metro.geo.str.contains('\\spart')]
dat_lf_metro = pd.concat([dat_lf_metro, dat_lf_metro.geo.str.split('\\,',1,True).rename(columns={0:'city',1:'prov'})],1).drop(columns=['geo'])
dat_lf_metro.lf = dat_lf_metro.lf.astype(object)
cn_sort = ['prov','city','lf']
assert np.all(dat_lf_metro.groupby(cn_sort+['year']).size()==1)
dat_lf_metro = dat_lf_metro.sort_values(cn_sort+['year'])
# dat_lf_metro = dat_lf_metro[dat_lf_metro.city.isin(list(di_city))]
dat_lf_metro = dat_lf_metro.drop(columns=['prov','year'])
dat_lf_metro = dat_lf_metro.assign(metro=lambda x: x.city.map(di_city).fillna('Other'))
dat_lf_metro = dat_lf_metro.groupby(['date','lf','metro']).value.sum().reset_index()
dat_lf_metro = dat_lf_metro.sort_values(['lf','metro','date']).reset_index(None,True)
# Change in the levels
dat_lf_metro = dat_lf_metro.assign(delta = lambda x: x.value-x.groupby(['lf','metro']).value.shift(1))

# CANADA LEVEL
dat_lf_can = sc.table_to_df('14-10-0090-01')
dat_lf_can.rename(columns=di_cn,inplace=True)
dat_lf_can = dat_lf_can[dat_lf_can.lf.isin(['Employment','Population']) & (dat_lf_can.geo=='Canada')][cn]
dat_lf_can['year'] = dat_lf_can.date.dt.strftime('%Y').astype(int)
dat_lf_can = dat_lf_can.sort_values(['lf','year']).reset_index(None,True)
# Change in the level
dat_lf_can = dat_lf_can.assign(delta=lambda x: x.value-x.groupby('lf').value.shift(1))

# Merge
cn_lf = ['date','lf','delta']
df_lf = dat_lf_can[cn_lf].merge(dat_lf_metro[cn_lf + ['metro']],'left',['date','lf']).dropna()
df_lf = df_lf.rename(columns={'delta_x':'can', 'delta_y':'city'}).assign(share = lambda x: x.city/x.can)
df_lf = df_lf.query('date>=@dstart').reset_index(None,True)

#########################################
### --- (5) Mortgage originations --- ###

di_mort = {'REF_DATE':'date','Categories':'tt','VALUE':'value'}
df_mort = sc.table_to_df('38-10-0238-01')
df_mort.rename(columns=di_mort,inplace=True)
df_mort = df_mort[list(di_mort.values())]
df_mort = df_mort[df_mort.tt.str.contains('^Mortgages')]
df_mort.date = pd.to_datetime(df_mort.date)
df_mort.tt = np.where(df_mort.tt.str.contains('flow'),'flow','stock')
df_mort = df_mort.query('tt=="stock"').reset_index(None,True)
df_mort = df_mort.assign(quarter=lambda x: x.date.dt.quarter,year=lambda x: x.date.dt.year)
# Quarterly growth in mortgages vs quarterly growth in HPI
tmp = df_tera.query('city=="Canada"').assign(quarter=lambda x: x.date.dt.quarter).rename(columns={'idx':'value'})
cn_left = ['year','quarter','value']
df_mort_tera = df_mort[cn_left].merge(tmp[cn_left],'left',cn_left[:-1],suffixes=('_mort','_tera')).dropna()
df_mort_tera = df_mort_tera.melt(cn_left[:-1],None,'tt').assign(tt=lambda x: x.tt.str.replace('value_',''))
df_mort_tera = df_mort_tera.groupby(['tt','year','quarter']).value.mean().reset_index()
df_mort_tera = df_mort_tera.assign(qq=lambda x: x.value/x.groupby('tt').value.shift(1)-1).dropna()
df_mort_tera = df_mort_tera.assign(date=lambda x: pd.to_datetime(x.year.astype(str)+'-'+(x.quarter*3).astype(str)))
df_mort_tera = idx_first(df_mort_tera.drop(columns=['year','quarter']), 'tt', 'date', 'value')
df_mort_tera = df_mort_tera.rename(columns={'value':'idx'}).melt(['tt','date'],None,'msr')

#########################
### --- (6) REITS --- ###

di_tt_reit = {'Office':'Commercial', 'Hotels':'Commercial', 'Diversified':'Both',
         'Residential':'Residential', 'Retail':'Commercial', 
         'Healthcare':'Commercial', 'Industrial':'Commercial'}

# (i) REIT list
lnk = 'https://raw.githubusercontent.com/ErikinBC/gists/master/data/reit_list.csv'
dat_reit = pd.read_csv(lnk,header=None).iloc[:,0:3]
dat_reit.columns = ['name', 'ticker', 'tt']
dat_reit = dat_reit.assign(ticker = lambda x: x.ticker.str.replace('.','-',regex=False) + '.TO')
# Remove certain REITS
drop_ticker = ['SRT-UN.TO', 'SMU-UN.TO']
dat_reit = dat_reit.query('~ticker.isin(@drop_ticker)',engine='python').reset_index(None,True)
di_ticker = {'FCD-UN.TO':'FCD-UN.V', 'FRO-UN.TO':'FRO-UN.V'}
dat_reit.ticker = [di_ticker[tick] if tick in di_ticker else tick  for tick in dat_reit.ticker]
smatch = 'REIT|Properties|Residences|Property|Trust|Industrial|Commercial|North American'
dat_reit['name2'] = dat_reit.name.str.replace(smatch,'',regex=True).str.strip().replace('\\s{2,}',' ',regex=True)
n_reit = dat_reit.shape[0]
print('Number of reits: %i' % n_reit)
# For merging
dat_reit2 = dat_reit[['ticker','name2','tt']].rename(columns={'name2':'name'})

# (ii) Load data
holder, holder_daily = [], []
for ii, rr in dat_reit.iterrows():
  name, ticker, tt = rr['name'], rr['ticker'], rr['tt']
  print('Stock: %s (%i of %i)' % (ticker, ii+1, n_reit))
  # (i) monthly price & dividends
  tmp_df = get_price_dividend(ticker, dstart, dnow)
  holder.append(tmp_df)
  # (ii) Get daily price data
  tmp_df_daily = get_price_dividend(ticker, dstart, dnow, ret_daily=True)
  holder_daily.append(tmp_df_daily)
# Monthly data - merge with the stock info
df_reit = pd.concat(holder).reset_index(None, True)
df_reit = dat_reit2.merge(df_reit.drop(columns=['year','month']))
# Daily data
df_reit_daily = pd.concat(holder_daily).reset_index(None, True)

################################
### --- (7) OTHER STOCKS --- ###

# Load the case-shiller price data
ticker_cs = 'SPCS20RSA'
df_cs = pdr.fred.FredReader(symbols=ticker_cs,start='2000-01-01').read()
df_cs = df_cs.reset_index().rename(columns={'DATE':'date',ticker_cs:'shiller'})

# Load in mortgage rates
df_rates = pd.read_csv('irates.csv')
df_rates.date = pd.to_datetime(df_rates.date)

# merge
df_shiller_mrate = df_cs.merge(df_rates,'left')

# Get stock list
dat_other = pd.read_csv('stock_list.csv')

# (ii) Load data
holder, holder_daily = [], []
for ii, rr in dat_other.iterrows():
  name, ticker = rr['name'], rr['ticker']
  print('Stock: %s (%i of %i)' % (ticker, ii+1, dat_other.shape[0]))
  # (i) monthly price & dividends
  tmp_df = get_price_dividend(ticker, dstart, dnow)
  holder.append(tmp_df)
  # (ii) daily price
  tmp_df_daily = get_price_dividend(ticker, dstart, dnow, ret_daily=True)
  holder_daily.append(tmp_df_daily)

# Merge
df_other = pd.concat(holder).reset_index(None,True)
df_other = dat_other.drop(columns='full').merge(df_other,'right')
# Daily
df_other_daily = pd.concat(holder_daily).reset_index(None,True)


########################
### --- (8) SAVE --- ###

di_storage = {'teranet': df_tera, 'tera_w':mm_cities, 'crea':df_crea, 
              'cpi_tsx':df_cpi_tsx, 'lf':df_lf, 'mort':df_mort_tera,
              'reit':df_reit, 'reit_daily':df_reit_daily,
              'shiller':df_shiller_mrate,
              'other':df_other, 'other_daily':df_other_daily}
with open('data_for_plots.pickle', 'wb') as handle:
    pickle.dump(di_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
