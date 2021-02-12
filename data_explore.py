"""
SCRIPT TO GENERATE THE HOUSE PRICE INDEX AND STATSCAN DEMOGRAPHIC DATA
"""

import os
import pandas as pd
import numpy as np
import plotnine
from plotnine import *
from datetime import datetime
# import shutil
# import requests

# # For custom packages
# pip install yahoofinancials
# pip install yahoofinance
# pip install git+https://github.com/ianepreston/stats_can
from yahoofinancials import YahooFinancials
import yahoofinance
from stats_can import StatsCan
sc = StatsCan()

from support_funs import makeifnot, add_date_int

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

dstart = '2001-01-01' # Begin analysis at 2001 (lines up with StatsCan)
ystart = int(pd.to_datetime(dstart).strftime('%Y'))
dnow = datetime.now().strftime('%Y-%m-%d')
print('Current date: %s' % dnow)

###############################
### --- (1) TERANET HPI --- ###

# Download Teranet data (assume Linux/WSL)
fn_url = 'House_Price_Index.csv'
url_tera = 'https://housepriceindex.ca/_data/' + fn_url
if fn_url in os.listdir():
    print('Removing existing CSV')
    os.remove(fn_url)
os.system('wget ' + url_tera + ' --no-check-certificate')

df_tera = pd.read_csv('House_Price_Index.csv',header=[0,1])
idx = pd.IndexSlice
df_tera.rename(columns={'Unnamed: 0_level_1':'Index'},level=1,inplace=True)
tmp = df_tera.loc[:,idx[:,'Sales Pair Count']].values[:,0]
df_tera = df_tera.loc[:,idx[:,'Index']]
df_tera.columns = df_tera.columns.droplevel(1)
df_tera.rename(columns={'Transaction Date':'date','c11':'canada'},inplace=True)
df_tera.date = pd.to_datetime(df_tera.date,format='%b-%Y')
dat_sales_tera = pd.DataFrame({'date':df_tera.date, 'sales':tmp})
# df_tera.query('date >= @dstart').melt('date',None,'city').assign(city=lambda x: x.city.str.split('_'))
df_tera = df_tera[df_tera.date >= pd.to_datetime(dstart)].melt('date',None,'city').assign(city=lambda x: x.city.str.split('_'))
df_tera.city = df_tera.city.apply(lambda x: x[-1])
df_tera = add_date_int(df_tera).drop(columns=['day'])
df_tera_wide = df_tera.pivot_table('value',['year','month','date'],'city').reset_index()

# Calculate for four example cities (VAN/TOR/CAL/MON)
city_lvls = ['calgary','toronto','montreal','vancouver']
city_lbls = ['Calgary','Toronto','Montreal','Vancouver']
df_tera_sample = df_tera_wide.melt(['date'],city_lvls)
df_tera_sample = df_tera_sample.merge(df_tera_sample.groupby('city').head(1)[['city','value']],'left',['city'])
df_tera_sample = df_tera_sample.assign(price = lambda x: x.value_x / x.value_y*100,
                    city=lambda x: pd.Categorical(x.city.str.capitalize(),city_lbls))

# # Calculate annual sales
# dat_sales_tera = dat_sales_tera[dat_sales_tera.date >= pd.to_datetime(dstart)].assign(sales=lambda x: x.sales.astype(int))
# dat_sales_tera = add_date_int(dat_sales_tera).drop(columns=['day','date'])
# dat_sales_tera.groupby('year').sales.mean().reset_index().assign(sales=lambda x: (x.sales*12).astype(int))


################################
### --- (2) STATSCAN CPI --- ###

# ---- (1.B) All-Items CPI (Statistics Canada) ---- #
cn_cpi = ['REF_DATE','VALUE','Products and product groups']
di_cpi = dict(zip(cn_cpi, ['date','cpi','products']))
df_cpi = sc.table_to_df('18-10-0006-01')[cn_cpi].rename(columns=di_cpi)
df_cpi.date = pd.to_datetime(df_cpi.date)
df_cpi = df_cpi.query('products=="All-items" & date >= @dstart').reset_index(None,True).drop(columns='products')
df_cpi = df_cpi.assign(cpi = lambda x: x.cpi / x.cpi[0] * 100)

# Visualize housing market vs CPI
tmp = pd.DataFrame({'date':pd.to_datetime(['2015-01-01']),'y':150, 'txt':'CPI'})
gg_tera = (ggplot(df_tera_sample, aes(x='date',y='price',color='city')) +
           geom_line() + theme_bw() + labs(y='Index (100 == 2001)') +
           ggtitle("Teranet HPI; CPI-All Items") +
           theme(axis_title_x=element_blank(), axis_text_x=element_text(angle=90)) +
           scale_color_discrete(name='City') +
           scale_x_datetime(breaks='3 years',date_labels='%Y') +
           geom_text(aes(x='date',y='y',label='txt'),data=tmp,inherit_aes=False) +
           geom_line(aes(x='date',y='cpi'),color='black',data=df_cpi))
gg_tera.save(os.path.join(dir_figures, 'gg_tera.png'),width=6, height=4)

#######################################
### --- (3) STATSCAN POPULATION --- ###

di_cn = {'GEO':'geo','Age group':'age','REF_DATE':'date',
                   'Labour force characteristics':'lf', 'VALUE':'value'}
cn = ['date','geo','value','lf']

# LOAD CITY-LEVEL
df_metro = sc.table_to_df('14-10-0096-01')
df_metro.rename(columns=di_cn,inplace=True)
df_metro = df_metro[(df_metro.Sex=='Both sexes') & (df_metro.age=='15 years and over') & df_metro.lf.isin(['Population','Employment'])][cn]
df_metro['year'] = df_metro.date.dt.strftime('%Y').astype(int)
# Remove the Quebec/Ontario part of Ottawa
df_metro = df_metro[~df_metro.geo.str.contains('\\spart')]
df_metro = pd.concat([df_metro, df_metro.geo.str.split('\\,',1,True).rename(columns={0:'city',1:'prov'})],1).drop(columns=['geo'])
df_metro.lf = df_metro.lf.astype(object)
cn_sort = ['prov','city','lf']
assert np.all(df_metro.groupby(cn_sort+['year']).size()==1)
df_metro = df_metro.sort_values(cn_sort+['year']).reset_index(None,True)
# Index to 2001
df_metro = df_metro.merge(df_metro.groupby(cn_sort).head(1).rename(columns={'value':'idx'})[cn_sort+['idx']],'left',cn_sort)
df_metro = df_metro.assign(delta = lambda x: x.value - x.idx)[cn_sort+['year','delta','value']].rename(columns={'value':'level'})
# Map city types
di_city = {'Vancouver':'Van/Vic','Victoria':'Van/Vic',
          'Calgary':'Cal/Edm','Edmonton':'Cal/Edm',
          'Montréal':'Montreal', 'Toronto':'GTAH','Hamilton':'GTAH'}
other_city = list(np.setdiff1d(df_metro.city.unique(),list(di_city)))
di_city = {**di_city,**dict(zip(other_city,np.repeat('other',len(other_city))))}
df_metro['city_comb'] = df_metro.city.map(di_city)

# Aggergate by year/city_comb
df_metro_comb = df_metro.groupby(['year','city_comb','lf'])[['delta','level']].sum().reset_index()

# LOAD AGGREGATE DATA
df_employ_agg = sc.table_to_df('14-10-0090-01')
df_employ_agg.rename(columns=di_cn,inplace=True)
df_employ_agg = df_employ_agg[df_employ_agg.lf.isin(['Employment','Population']) & (df_employ_agg.geo=='Canada')][cn]
df_employ_agg['year'] = df_employ_agg.date.dt.strftime('%Y').astype(int)
df_employ_agg = df_employ_agg.sort_values(['lf','year']).reset_index(None,True)
# Index to 2001
df_employ_agg = df_employ_agg.merge(df_employ_agg.groupby('lf').head(1).rename(columns={'value':'idx'})[['lf','idx']],'left')
df_employ_agg = df_employ_agg.assign(delta = lambda x: x.value - x.idx)[['year','delta','lf','value']].rename(columns={'value':'level'})
# Index the cities against national total
df_employ = df_metro_comb.merge(df_employ_agg,'left',['year','lf'],suffixes=('_city','_cad')).melt(['year','city_comb','lf'],None,'tmp')
df_employ = df_employ.assign(metric=lambda x: x.tmp.str.split('_',1,True).iloc[:,0],
                 geo=lambda x: x.tmp.str.split('_',1,True).iloc[:,1]).drop(columns='tmp')
df_employ = df_employ.pivot_table('value',['year','city_comb','lf','metric'],'geo').reset_index().sort_values(['year','lf','metric','city_comb'])
df_employ = df_employ.query('year > @ystart').reset_index(None,True).assign(share = lambda x: x.city / x.cad)

# Figure: Greater Toronto leads in Canadian job creation
tmp = df_employ.query('((metric=="delta" & lf=="Employment") | '
                '(metric=="level" & lf=="Population")) & city_comb != "other"')
gg_emp = (ggplot(tmp,aes(x='year',y='share',color='city_comb')) +
          theme_bw() + geom_point() + geom_line() + facet_wrap('~lf') +
          labs(y='Share of net job creation/population levels') +
          ggtitle('') +
          theme(axis_title_x=element_blank(), legend_position='bottom', legend_box_spacing=0.2) +
          scale_color_discrete(name='City'))
gg_emp.save(os.path.join(dir_figures,'gg_emp.png'),width=9,height=5)

#########################################
### --- (4) Mortgage originations --- ###

di_mort = {'REF_DATE':'date','Categories':'tt','VALUE':'value'}
df_mort = sc.table_to_df('38-10-0238-01')
df_mort.rename(columns=di_mort,inplace=True)
df_mort = df_mort[list(di_mort.values())]
df_mort = df_mort[df_mort.tt.str.contains('^Mortgages')]
df_mort.date = pd.to_datetime(df_mort.date)
df_mort.tt = np.where(df_mort.tt.str.contains('flow'),'flow','stock')
df_mort = df_mort.sort_values(['tt','date']).reset_index(None,True)
# Note the difference in stock does not equal flow: determine why flow!=stock
df_mort = df_mort.pivot('date','tt','value').reset_index()

# Mortgage originations vs sales and HPI

#########################################
### --- (5) Housing GDP --- ###


#########################################
### --- (6) Housing resales --- ###

# Toronto: Annual (https://trreb.ca/files/market-stats/market-watch/historic.pdf)
#  Monthly (https://trreb.ca/index.php/market-news/market-watch/market-watch-archive)

# Vancouver (https://www.rebgv.org/market-watch/monthly-market-report.html), 1999
# Fraser Valley (http://www.fvreb.bc.ca/stats/), 2008

# Winnipeg (https://www.winnipegregionalrealestateboard.ca/market-statistics/archived-market-releases?bulletin3%5Bpage%5D=16), 2010

# Montreal: (https://apciq.ca/en/real-estate-market/statistiques-mensuelles-detaillees/) 2013

# Ottawa (https://www.oreb.ca/wp-content/uploads/2017/03/OREBNewsRelease-March2016.pdf) 2016

# Need to contact CREB, RAE, Nova scotia association of realtors



#########################
### --- (7) REITS --- ###

# ---- (1.C) REIT list ---- #
lnk = 'https://raw.githubusercontent.com/ErikinBC/gists/master/data/reit_list.csv'
df_reit = pd.read_csv(lnk,header=None).iloc[:,0:3]
df_reit.columns = ['name', 'ticker', 'tt']
df_reit = df_reit.assign(ticker = lambda x: x.ticker.str.replace('.','-') + '.TO')
di_ticker = {'FCD-UN.TO':'FCD-UN.V', 'FRO-UN.TO':'FRO-UN.V' ,'NXR-UN.TO':'NXR-UN.V'}
df_reit.ticker = [di_ticker[tick] if tick in di_ticker else tick  for tick in df_reit.ticker]
smatch = 'REIT|Properties|Residences|Property|Trust|Industrial|Commercial|North American'
df_reit['name2'] = df_reit.name.str.replace(smatch,'').str.strip().replace('\\s{2,}',' ')
print(df_reit.shape)

# ---- (1.D) Load in the Yahoo Finance data --- #
holder = []
for ii, rr in df_reit.iterrows():
  name, ticker, tt = rr['name'], rr['ticker'], rr['tt']
  print('Stock: %s (%i of %i)' % (ticker, ii+1, df_reit.shape[0]))
  stock = YahooFinancials(ticker)
  # (1) Get monthly price
  cn_keep = ['formatted_date','open']
  monthly = stock.get_historical_price_data(dstart, dnow, 'monthly')[ticker]
  tmp = pd.DataFrame(monthly['prices'])[cn_keep].rename(columns={'formatted_date':'date','open':'price'})
  tmp = tmp.assign(name=name, ticker=ticker, tt=tt, date=lambda x: pd.to_datetime(x.date))
  tmp = add_date_int(tmp)
  tmp = tmp[tmp.day == 1].drop(columns=['day'])
  # (2) Dividend info
  tmp2 = pd.DataFrame(stock.get_daily_dividend_data(dstart, dnow)[ticker]).drop(columns=['date'])
  tmp2 = tmp2.rename(columns={'formatted_date':'date','amount':'dividend'}).assign(date=lambda x: pd.to_datetime(x.date))
  tmp2 = tmp2.assign(year=lambda x: x.date.dt.strftime('%Y').astype(int), month=lambda x: x.date.dt.strftime('%m').astype(int)).drop(columns=['date'])
  # Merge with price
  tmp3 = tmp.merge(tmp2,'left',['year','month'])
  tmp3.price = tmp3.price.fillna(method='backfill')
  # (3) Balance sheet info
  bs = yahoofinance.BalanceSheet(ticker).to_dfs()
  tmp_lia = bs['Liabilities'].reset_index()
  tmp_lia = tmp_lia[tmp_lia.Item == 'Total Liabilities'].melt('Item',None,'date','liability').drop(columns=['Item'])
  tmp_asset = bs['Assets'].reset_index()
  tmp_asset = tmp_asset[tmp_asset.Item == 'Total Assets'].melt('Item',None,'date','asset').drop(columns=['Item'])
  tmp_balance = tmp_asset.merge(tmp_lia)
  tmp4 = tmp_balance.assign(year=lambda x: pd.to_datetime(x.date).dt.strftime('%Y').astype(int)).drop(columns=['date'])
  # Merge
  tmp5 = tmp3.merge(tmp4,'left','year')
  holder.append(tmp5)

df = pd.concat(holder).reset_index(None, True)

# Merge the data
cn_wide = ['year','month','canada','toronto']
df = df.merge(df_tera_wide[cn_wide],'left',['year','month'])

# Go with the minimum dividend if duplicated
tmp = df[['year','month','name','dividend']].copy()
tmp.insert(tmp.shape[1],'cidx', tmp.groupby(['year','month','name']).cumcount())
tmp = tmp.groupby(['year','month','name']).dividend.min().reset_index()
df = df.drop(columns=['dividend']).merge(tmp,'left',['year','month','name'])

# Remove any duplicated dividend payout info
df = df[~df[['year','month','name','dividend']].duplicated()].reset_index(None, True)

assert df.groupby(['year','name']).size().max() == 12

# # Save snapshot of data?
# should_save = True
# path = os.path.join(dir_base,'df_'+dnow.replace('-','_')+'.csv')
# if should_save:
#   df.to_csv(path, index=False)

# # Load existing file
# fn_load = pd.Series(os.listdir(dir_base))
# fn_load = fn_load[fn_load.str.contains('df_[0-9]')].reset_index(None,True)
# fn_load = fn_load[pd.to_datetime(fn_load.str.split('_',expand=True,n=1).iloc[:,1].str.replace('.csv',''),format='%Y_%m_%d').idxmax()]
# df = pd.read_csv(os.path.join(dir_base,fn_load))

# ---- (1.E) Calculate the dividend rate --- #
di_tt = {'Office':'Commercial', 'Hotels':'Commercial', 'Diversified':'Both',
         'Residential':'Residential', 'Retail':'Commercial', 'Healthcare':'Commercial', 'Industrial':'Commercial'}

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

# BALANCE SHEET

equity = df[df.asset.notnull()].groupby(['year','name'])[['asset','liability']].mean().reset_index()
equity = equity.assign(eshare=lambda x: 1 - x.liability/x.asset).merge(df_reit).assign(tt2 = lambda x: x.tt.map(di_tt))
equity.name2 = pd.Categorical(equity.name2,equity.groupby('name2').eshare.mean().reset_index().sort_values('eshare',ascending=False).name2)
equity.head()

plotnine.options.figure_size = (14,8)

g2 = ggplot(equity, aes(x='year',y='eshare',fill='tt2')) + geom_bar(stat='identity') + \
  theme_bw() + scale_fill_discrete(name=' ') + facet_wrap('~name2',ncol=8) + \
  ggtitle('Share of equity') + geom_hline(yintercept=equity.eshare.mean(),color='black') + \
  theme(axis_title_x=element_blank()) + labs(y='Equity share')
g2

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