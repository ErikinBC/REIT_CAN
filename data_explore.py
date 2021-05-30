"""
SCRIPT TO GENERATE THE HOUSE PRICE INDEX AND STATSCAN DEMOGRAPHIC DATA
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

from stats_can import StatsCan
sc = StatsCan()
import pickle

from support_funs import add_date_int, get_YahooFinancials, idx_first, rm_if_exists

dir_base = os.getcwd()

# Begin analysis at 2005 for CREA (lines up with StatsCan)
dstart = '2005-01-01'
dstart_dt = pd.to_datetime(dstart)
ystart = int(pd.to_datetime(dstart).strftime('%Y'))
dnow = datetime.now().strftime('%Y-%m-%d')
print('Current date: %s' % dnow)


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

# (ii) Load data
holder = []
for ii, rr in dat_reit.iterrows():
  name, ticker, tt = rr['name'], rr['ticker'], rr['tt']
  print('Stock: %s (%i of %i)' % (ticker, ii+1, n_reit))
  # (i) monthly price & dividends
  tmp_df = get_YahooFinancials(ticker, dstart, dnow)
  holder.append(tmp_df)
df_reit = pd.concat(holder).reset_index(None, True)
df_reit.dividend = df_reit.dividend.fillna(0)

# (iii) Clean up dividend history
df_reit.groupby(['ticker','year']).dividend.sum().reset_index().query('dividend==0')


qq = df_reit.groupby(['ticker','year']).apply(lambda x: 
  pd.Series({'price':x.price.mean(),'dividend':x.dividend.sum()}))
qq = qq.assign(pct=lambda x: x.dividend/x.price).reset_index()
qq.sort_values('pct',ascending=False)

qq.query('ticker == "KMP-UN.TO"')


# ---- (1.E) Calculate the dividend rate --- #


ann_dividend = df.groupby(['year','name']).dividend.apply(lambda x:
      pd.Series({'mu':x.mean(),'n':len(x),'null':x.isnull().sum()})).reset_index()
ann_dividend = ann_dividend.pivot_table('dividend',['year','name'],'level_2').reset_index().sort_values(['name','year']).reset_index(None,True)
ann_dividend[['n','null']] = ann_dividend[['n','null']].astype(int)
ann_dividend = ann_dividend.assign(neff = lambda x: x.n - x.null)
tmp = ann_dividend[ann_dividend.neff >= 3].groupby(['name','year']).apply(lambda x: 12 * x.mu).reset_index().rename(columns={'mu':'adiv'}) # * (12/(x.n-x.null))
ann_dividend = ann_dividend.merge(tmp,'left',['name','year'])[['name','year','adiv']].sort_values(['name','year']).reset_index(None,True)
# Price to dividend ratio
ann_dividend = ann_dividend.merge(df.groupby(['year','name']).price.mean().reset_index()).assign(rate=lambda x: x.adiv / x.price)
ann_dividend = ann_dividend.merge(df_reit,'left','name')  #.drop(columns=['ticker'])


tmp = ann_dividend[(ann_dividend.rate < 0.2) & (ann_dividend.year >= 2005)].assign(tt = lambda x: x.tt.map(di_tt))
tmp2 = tmp.groupby('name2').rate.mean().reset_index()
# Get order
tmp.name2 = pd.Categorical(tmp.name2,ann_dividend.groupby('name2').rate.mean().reset_index().sort_values('rate',ascending=False).name2)
plotnine.options.figure_size = (16,10)
g1 = (ggplot(tmp.sort_values('name2'),aes(x='year',y='rate',color='tt')) + geom_point()  + geom_line() +
  geom_hline(yintercept=ann_dividend.rate.mean(),color='black') + facet_wrap('~name2',ncol=8) + theme_bw() +
  ggtitle('Annualized dividend rates since 2005') + scale_y_continuous(limits=[0,0.2],breaks=np.arange(0,0.21,0.05)) +
  scale_color_discrete(name=' '))
g1


### (4) HOUSE PRICE TRACKING ###
hp = df.melt(id_vars=['year','month','price','ticker'],value_vars=['canada','toronto'],var_name='city',value_name='hp')
# Annualized change
hp = hp[hp.hp.notnull()]
rho = hp.groupby(['ticker','city']).apply(lambda x: np.corrcoef(x.price, x.hp)[0,1]).reset_index().rename(columns={0:'rho'})
rho = rho.merge(df_reit).assign(tt2 = lambda x: x.tt.map(di_tt), ticker2= lambda x: x.ticker.str.split('-',1,True).iloc[:,0])
rho.name2 = pd.Categorical(rho.name2, rho.groupby('name2').rho.mean().reset_index().sort_values('rho',ascending=False).name2)

plotnine.options.figure_size = (10,5)
g3 = ggplot(rho,aes(x='name2',y='rho',color='tt2',shape='city')) + geom_point() + \
  ggtitle('Correlation to House Price Index') + labs(y='Correlation') + \
  theme_bw() + theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + \
  geom_hline(yintercept=0) + scale_shape_discrete(name=' ',labels=['Canada','Toronto']) + \
  scale_color_discrete(name=' ')
g3

### (5) RANKING ACROSS FACTORS

# Average of all annual correlations for Tor+CAD
rank_rho = rho.groupby('name').rho.mean().reset_index().sort_values('rho',ascending=False).reset_index(None,True)
rank_rho = rank_rho.assign(rank=np.arange(rank_rho.shape[0])+1,tt='rho').rename(columns={'rho':'metric'})
# Last year's equity ratio
rank_equity = equity[equity.year == equity.year.max()][['name','eshare']].sort_values('eshare',ascending=False).reset_index(None,True)
rank_equity = rank_equity.assign(rank=np.arange(rank_equity.shape[0])+1,tt='eshare').rename(columns={'eshare':'metric'})
# Last 3 years average dividends
rank_dividend = ann_dividend[(ann_dividend.year >= (ann_dividend.year.max()-3)) & (ann_dividend.year < ann_dividend.year.max())]
rank_dividend = rank_dividend.groupby('name').rate.mean().reset_index().sort_values('rate',ascending=False).reset_index(None,True)
rank_dividend = rank_dividend.assign(rank=np.arange(rank_dividend.shape[0])+1,tt='dividend').rename(columns={'rate':'metric'})
# Merge
rank_all = pd.concat([rank_rho, rank_equity, rank_dividend],0).pivot('name','tt','rank').reset_index()
rank_all = df_reit.merge(rank_all).assign(tt2 = lambda x: x.tt.map(di_tt))
w_dividend, w_eshare, w_rho = 0.1, 0.3, 0.6
rank_all = rank_all.assign(total = lambda x: (w_dividend*x.dividend + w_eshare*x.eshare + w_rho*x.rho)/(w_dividend+w_eshare+w_rho)).sort_values('total')
rank_all = rank_all.reset_index(None,True).assign(total = lambda x: np.round(x.total,1))
rank_all.name2 = pd.Categorical(rank_all.name2, rank_all.name2[::-1])
rank_all_long = rank_all.melt(['name2','tt2'],['dividend','eshare','rho','total'],'rank')

plotnine.options.figure_size = (8,8)
g3 = ggplot(rank_all_long,aes(y='name2',x='value',color='rank',shape='tt2')) + geom_point(size=3) + \
        scale_color_manual(name='Metric',labels=['Dividend','Equity','Correlation','Total'],values=["#F8766D","#00BA38","#619CFF",'black']) + \
        theme_bw() + theme(axis_title_y=element_blank()) + \
        scale_shape_manual(name='Type',values=['$B$','$C$','$R$']) + \
        labs(x='Rank') + ggtitle('Final Rank of REITs')
g3

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
df_tsx = get_YahooFinancials(ticker_tsx,dstart,dnow,dividend=False)
df_tsx = df_tsx.drop(columns=['ticker','year','month']).rename(columns={'price':'tsx'})
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
### --- (6) SAVE FOR LATER --- ###


di_storage = {'teranet': df_tera, 'tera_w':mm_cities, 'crea':df_crea, 
              'cpi_tsx':df_cpi_tsx, 'lf':df_lf, 'mort':df_mort_tera}
with open('data_for_plots.pickle', 'wb') as handle:
    pickle.dump(di_storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
