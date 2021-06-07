# Load libraries
import numpy as np
import pandas as pd
import os
from time import time
from plotnine import *labels, 
from mizani.formatters import percent_format
import pickle

from statsmodels.stats.proportion import proportion_confint as prop_CI

from funs_support import add_date_int, makeifnot, gg_save, ym2date, gg_color_hue, idx_first
from funs_stats import make_index

dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')
makeifnot(dir_figures)

with open('data_for_plots.pickle', 'rb') as handle:
    di_storage = pickle.load(handle)

df_tera = di_storage['teranet']
tera_w = ym2date(di_storage['tera_w'])
df_crea = di_storage['crea']
df_cpi_tsx = di_storage['cpi_tsx']
df_lf = di_storage['lf']
df_mort_tera = di_storage['mort']
df_reit = di_storage['reit']
df_shiller_mrate = di_storage['shiller']
df_other = di_storage['other']

assert np.issubdtype(df_cpi_tsx.date,np.datetime64)
dstart = df_tera.date.min().strftime('%Y-%m-%d')

# Concert REIT types
df_reit = df_reit.assign(tt=lambda x: np.where(x.tt.isin(['Diversified','Residential']),x.tt,'Commerical'))

# Subset canada
tera_cad = df_tera.query('city=="Canada"')[['date','idx']]

###############################
# -- (5) TERANET + SHILLER -- #

# .drop(columns='variable')
tmp = df_shiller_mrate.rename(columns={'fixed':'fyf','variable':'fyv'}).melt('date',None,'msr','idx').dropna().reset_index(None,True)
df_shiller_idx = idx_first(tmp, 'msr', 'date', 'idx')#.pivot('date','variable','value').reset_index()
df_shiller_idx = df_shiller_idx.assign(mm=lambda x: x.idx/x.groupby('msr').idx.shift(1)-1)
df_shiller_idx = tera_cad.assign(mm=lambda x: x.idx/x.idx.shift(1)-1).merge(df_shiller_idx,'left','date',suffixes=('_tera','_shiller')).sort_values(['msr','date'])
df_shiller_idx = df_shiller_idx.melt(['msr','date'],None,'tmp').dropna()
df_shiller_idx['tt'] = df_shiller_idx.tmp.str.split('\\_',1,True).iloc[:,0]
df_shiller_idx['tmp2'] = df_shiller_idx.tmp.str.split('\\_',1,True).iloc[:,1]
df_shiller_idx = df_shiller_idx.drop(columns='tmp').pivot_table('value',['msr','tt','date'],'tmp2').reset_index()
# Split into mrate and index
shiller_mm = df_shiller_idx.query('tt != "idx"').dropna().drop(columns='tt').reset_index(None,True).assign(year=lambda x: x.date.dt.year)
shiller_idx = idx_first(df_shiller_idx.query('tt == "idx"'), 'msr', 'date', 'tera').drop(columns='tt').melt(['date','msr'],None,'tt')
di_msr = {'fyf':'5-year fixed', 'fyv':'5-year variable', 'shiller':'Case-Shiller 20 city'}
# Compare
gg_shiller_mm = (ggplot(shiller_mm,aes(x='shiller',y='tera',color='year')) + 
    theme_bw() + labs(x='m/m',y='Teranet (m/m)') + 
    geom_point() + theme(subplots_adjust={'wspace': 0.15}) + 
    ggtitle('Month-on-month growth rates (%)') + 
    geom_hline(yintercept=0,linetype='--') + 
    geom_vline(xintercept=0,linetype='--') + 
    facet_wrap('~msr',labeller=labeller(msr=di_msr),scales='free'))
gg_save('gg_shiller_mm.png', dir_figures, gg_shiller_mm, 14, 3.5)

# Compare
gg_shiller_idx = (ggplot(shiller_idx,aes(x='date',y='value',color='tt')) + 
    theme_bw() + geom_line() + labs(y='Index') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m') + 
    scale_color_discrete(name='Measure',labels=['Facet','Teranet']) + 
    facet_wrap('~msr',labeller=labeller(msr=di_msr),scales='free_x'))
gg_save('gg_shiller_idx.png', dir_figures, gg_shiller_idx, 13, 3.75)


##########################
# -- (6) OTHER STOCKS -- #

other_names = df_other.groupby(['name','ticker']).size().reset_index().drop(columns=[0])

price_other = df_other[['ticker','date','year','price','dividend']].copy()
# Get annual dividend rate
drate_other = price_other.groupby(['ticker','year']).apply(lambda x: 
    pd.DataFrame({'n':len(x),'p':x.price.mean(),'d':x.dividend.sum()},index=[0])).reset_index().drop(columns='level_2')
drate_other = drate_other.assign(rate=lambda x: x.d*(12/x.n)/x.p)
drate_other = drate_other.assign(cgain=lambda x: x.p/x.groupby('ticker').p.shift(1)-1)
drate_other = drate_other.melt(['ticker','year'],['rate','cgain'],'tt')
drate_other = drate_other.dropna().merge(other_names)

lblz = ['$'+str(z+1)+'$' for z in range(len(other_names))]
di_tt = {'rate':'Dividend Rate','cgain':'Price change'}
gg_drate_other = (ggplot(drate_other,aes(x='year',y='value',color='name',shape='name')) +
    scale_shape_manual(name='Stock',values=lblz) + 
    theme_bw() + geom_point(size=3) + geom_line() + 
    theme(subplots_adjust={'wspace': 0.2}) + 
    geom_hline(yintercept=0,linetype='--') + 
    scale_color_discrete(name='Stock') + 
    labs(y='Dividend/capital gains rate',x='Date') + 
    facet_wrap('~tt',labeller=labeller(tt=di_tt),scales='free_y'))
gg_save('gg_drate_other.png', dir_figures, gg_drate_other, 11, 4)

# Index to Teranet and compare
tera_other = price_other.drop(columns=['dividend','year']).merge(tera_cad).sort_values(['ticker','date'])
tera_other = idx_first(tera_other, 'ticker', 'date', 'idx')
tera_other = idx_first(tera_other, 'ticker', 'date', 'price')
tera_other = tera_other.rename(columns={'idx':'Teranet','price':'Stock'}).melt(['ticker','date'],None,'tt')
tera_other = tera_other.merge(other_names)

gg_tera_other = (ggplot(tera_other,aes(x='date',y='value',color='tt')) + 
    theme_bw() + geom_line() + 
    facet_wrap('~name',nrow=2) + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m') + 
    scale_color_discrete(name='Measure'))
gg_save('gg_tera_other.png', dir_figures, gg_tera_other, 16, 7)

# Calculate monthly
tera_other_mm = tera_other.assign(mm=lambda x: x.value/x.groupby(['ticker','tt']).value.shift(1)-1).pivot_table('mm',['ticker','name','date'],'tt').reset_index().assign(year=lambda x: x.date.dt.year)

gg_tera_other_mm = (ggplot(tera_other_mm,aes(x='Stock',y='Teranet',color='year')) + 
    theme_bw() + geom_point() + 
    geom_hline(yintercept=0,linetype='--') + 
    geom_vline(xintercept=0,linetype='--') + 
    theme(subplots_adjust={'wspace': 0.2,'hspace':0.2}) + 
    facet_wrap('~name',nrow=2,scales='free'))
gg_save('gg_tera_other_mm.png', dir_figures, gg_tera_other_mm, 16, 7)

# Quadrant
tera_other_quad = tera_other_mm.copy().query('year>=2010')
tera_other_quad[['Stock','Teranet']] = tera_other_quad[['Stock','Teranet']].apply(lambda x: np.where(x>0,1,-1))
tera_other_quad = tera_other_quad.groupby(['name','Stock','Teranet']).size().reset_index().rename(columns={0:'n'})
tera_other_quad = tera_other_quad.merge(tera_other_quad.groupby(['name','Teranet']).n.sum().reset_index(),'left',['name','Teranet'],suffixes=('','_tera')).assign(sens_tera=lambda x: x.n/x.n_tera)
tera_other_quad = tera_other_quad.merge(tera_other_quad.groupby(['name','Stock']).n.sum().reset_index(),'left',['name','Stock'],suffixes=('','_stock')).assign(prec_stock=lambda x: x.n/x.n_stock)
tera_other_quad = tera_other_quad.query('Stock==Teranet').drop(columns=['Teranet','n']).melt(['name','Stock'],None,'tmp')
tera_other_quad = pd.concat([tera_other_quad.drop(columns='tmp'),
    tera_other_quad.tmp.str.split('\\_',1,True).rename(columns={0:'tmp',1:'tt'})],1)
tera_other_quad = tera_other_quad.pivot_table('value',['name','Stock','tt'],'tmp').reset_index()
tera_other_quad = tera_other_quad.assign(tt=lambda x: np.where(x.tt=='stock','prec','sens'),
    value=lambda x: np.where(x.tt=='prec',x.prec,x.sens),
    n=lambda x: x.n.astype(int)).drop(columns=['prec','sens'])
tera_other_quad  = tera_other_quad.assign(lb=lambda x: prop_CI(x.value*x.n,x.n,method='beta')[0], ub=lambda x: prop_CI(x.value*x.n,x.n,method='beta')[1])

di_Stock = {'-1':'Negative','1':'Positive'}
posd = position_dodge(0.5)
gg_tera_quad = (ggplot(tera_other_quad,aes(x='name',y='value',color='tt')) + 
    theme_bw() + geom_point(position=posd) + 
    ggtitle('Since 2010') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    geom_linerange(aes(ymin='lb',ymax='ub'),position=posd) + 
    scale_color_discrete(name='Measure',labels=['Precision','Sensitivty']) + 
    facet_wrap('~Stock',labeller=labeller(Stock=di_Stock)))
gg_save('gg_tera_quad.png', dir_figures, gg_tera_quad, 10, 4)



############################
# -- (1) TERANET + CREA -- #

# (i) Teranet: weighted sales != aggregate
gg_tera_w = (ggplot(tera_w,aes(x='date',y='idx*100',color='tt')) + 
    geom_line() + theme_bw() + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    labs(y='Index (2001M1==100)') + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m') + 
    scale_color_discrete(name='Aggregate',labels=['Canada','Weighted Cities']))
gg_save('gg_tera_w.png',dir_figures,gg_tera_w,5,4)

# (ii) CREA vs Teranet level
cn_left = ['date','city','tt']
df_hpi_both = df_crea.merge(df_tera[df_crea.columns],'left',cn_left).query('tt=="aggregate"')
df_hpi_both.rename(columns={'idx_x':'crea','idx_y':'tera'}, inplace=True)
df_hpi_both = df_hpi_both.melt(cn_left,None,'hpi').drop(columns='tt')

gg_tera_crea_lvl = (ggplot(df_hpi_both,aes(x='date',y='value',color='hpi')) + 
    geom_line() + theme_bw() + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    labs(y='Index (2001M1==100)') + 
    facet_wrap('~city',nrow=2) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m') + 
    scale_color_discrete(name='HPI',labels=['CREA','Teranet']))
gg_save('gg_tera_crea_lvl.png',dir_figures,gg_tera_crea_lvl,12,5)

# (iii) CREA vs Teranet m/m
tmp = df_hpi_both.groupby(['city','hpi']).apply(lambda x: x.value/x.value.shift(1)-1).reset_index()
df_hpi_w = df_hpi_both.assign(mm=tmp.sort_values('level_2').value.values)
df_hpi_w = df_hpi_w.pivot_table('mm',['date','city'],'hpi').reset_index()
df_hpi_w = df_hpi_w.sort_values(['city','date']).reset_index(None,True)

gg_tera_crea_pct = (ggplot(df_hpi_w,aes(x='crea',y='tera')) + 
    geom_point(size=0.5) + theme_bw() + 
    theme(axis_text_x=element_text(angle=90)) + 
    scale_x_continuous(labels=percent_format()) + 
    scale_y_continuous(labels=percent_format()) +  
    labs(x='CREA',y='Teranet',title='month-over-month %') + 
    facet_wrap('~city',nrow=2))
gg_save('gg_tera_crea_pct.png',dir_figures,gg_tera_crea_pct,12,5)

# (iv) Find the optimal correlation
lag_seq = np.arange(13)
alpha = 0.1
n_bs = 1000
holder = []
for lag in lag_seq:
    print(lag)
    tmp_lag = df_hpi_w.assign(crea=df_hpi_w.groupby('city').crea.shift(lag)).dropna()
    tmp_lag_bs = tmp_lag.groupby('city').sample(frac=n_bs,replace=True,random_state=n_bs)
    tmp_lag_bs['bidx'] = tmp_lag_bs.groupby('city').cumcount() % n_bs
    tmp_lag_bs = tmp_lag_bs.groupby(['city','bidx']).corr().reset_index()
    tmp_lag = tmp_lag.groupby('city').corr().reset_index()
    tmp_lag_bs = tmp_lag_bs.query('hpi=="crea"').drop(columns=['hpi','crea'])
    tmp_lag = tmp_lag.query('hpi=="crea"').drop(columns=['hpi','crea']).rename(columns={'tera':'rho'})
    tmp_lag_bs = tmp_lag_bs.groupby('city').tera.quantile([alpha/2,1-alpha/2]).reset_index()
    tmp_lag_bs = tmp_lag_bs.pivot('city','level_1','tera').rename(columns={alpha/2:'lb',1-alpha/2:'ub'})
    tmp_lag = tmp_lag.merge(tmp_lag_bs.reset_index()).assign(lag=lag)
    holder.append(tmp_lag)
df_rho_hpi = pd.concat(holder).reset_index(None,True)
df_rho_max = df_rho_hpi.loc[df_rho_hpi.groupby('city').rho.idxmax()]

gg_rho_hpi = (ggplot(df_rho_hpi,aes(x='lag',y='rho')) + theme_bw() + 
    geom_point() + geom_line() + 
    geom_linerange(aes(ymin='lb',ymax='ub')) + 
    labs(y='cor(Teranet,CREA[t-lag])',x='lag') + 
    ggtitle('Vertical lines show 90% CI\nRed number shows optimal CREA lag') + 
    facet_wrap('~city',nrow=2) + 
    geom_text(aes(label='lag'),data=df_rho_max,color='red',nudge_y=0.1) + 
    geom_hline(yintercept=0,linetype='--',color='blue') + 
    scale_x_continuous(breaks=list(range(0,13,2))))
    # scale_y_continuous(limits=[-0.25,1.0],breaks=list(np.arange(-0.25,1.01,0.25))))
gg_save('gg_rho_hpi.png',dir_figures,gg_rho_hpi,12,5)

# (v) CREA composition
colz = ['black'] + gg_color_hue(3)
lblz = ['Aggregate','Apartment','Townhouse','Single-Family']

gg_crea_tt = (ggplot(df_crea,aes(x='date',y='idx',color='tt')) + 
    geom_line() + theme_bw() + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    labs(y='CREA HPI (2001M1==100)') + 
    facet_wrap('~city',nrow=2) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m') + 
    scale_color_manual(name='Type',values=colz,labels=lblz))
gg_save('gg_crea_tt.png',dir_figures,gg_crea_tt,12,5)

###########################
# -- (2) HPI & ECONOMY -- #

df_hpi_vs_stats = tera_cad.merge(df_cpi_tsx,'left')
df_hpi_vs_stats = df_hpi_vs_stats.melt('date',None,'tt')
df_hpi_vs_stats = df_hpi_vs_stats.assign(tt=lambda x: pd.Categorical(x.tt,['idx','tsx','cpi']))

# (i) Housing vs Stock Market vs CPI
colz = ['black'] + gg_color_hue(2)
lblz = ['Teranet','TSX','CPI']
gg_hpi_vs_stats = (ggplot(df_hpi_vs_stats, aes(x='date',y='value',color='tt')) +
           geom_line() + theme_bw() + 
           labs(y='Index (2001M1==100)') + 
           theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
           scale_color_manual(name='Measure',values=colz,labels=lblz) + 
           scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m'))
gg_save('gg_hpi_vs_stats.png',dir_figures,gg_hpi_vs_stats,5,4)


# (ii) Employment share
tmp = df_lf.query('metro.isin(["GTAH","Van/Vic"])',engine='python')
gg_lf_share = (ggplot(tmp,aes(x='date',weight='share',fill='metro')) +
          theme_bw() + facet_wrap('~lf') + geom_bar(color='black') + 
          scale_y_continuous(limits=[-0.01,1]) + 
          labs(y='Share of annual net change') + 
          scale_x_datetime(date_breaks='5 years',date_labels='%Y') + 
          theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
          scale_fill_discrete(name='Metro'))
gg_save('gg_lf_share.png',dir_figures,gg_lf_share,9,5)

# (iii) Morgage growth and house price
di_msr = {'qq':'Q/Q %','idx':'2005Q2=100'}
gg_mort_tera =(ggplot(df_mort_tera,aes(x='date',y='value',color='tt')) + 
    theme_bw() + geom_line() + 
    theme(subplots_adjust={'wspace': 0.25}) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45),
            subplots_adjust={'wspace': 0.15}) + 
    scale_color_discrete(name='Measure',labels=['Mortgages','Teranet']) + 
    facet_wrap('~msr',scales='free',labeller=labeller(msr=di_msr)) + 
    labs(y='Mortgage & HPI growth/levels'))
gg_save('gg_mort_tera.png',dir_figures,gg_mort_tera,9,4)



###################
# -- (3) REITS -- #

dat_reit = df_reit.groupby(['ticker','name','tt']).size().reset_index().drop(columns=[0])
dat_reit.tt.value_counts()

# Annualized dividend rate
dividend = add_date_int(df_reit).groupby(['ticker','year']).apply(lambda x: 
  pd.Series({'price':x.price.mean(),'dividend':x.dividend.sum(),'n':len(x)}))
dividend = dividend.reset_index().assign(dividend=lambda x: x.dividend*(12/x.n))
dividend = dividend.assign(pct=lambda x: x.dividend/x.price)
# Group into quartiles of average rate
tmp_qq = dividend.groupby('ticker').pct.mean().sort_values()
qq_seq = ['q4','q3','q2','q1']
di_qq = dict(zip(qq_seq,['Q4','Q3','Q2','Q1']))
tmp_qq = pd.qcut(tmp_qq,4,labels=qq_seq).reset_index().rename(columns={'pct':'q4'})
tmp_qq.q4 = pd.Categorical(tmp_qq.q4,np.sort(qq_seq))
dividend = dividend.merge(tmp_qq).merge(dat_reit)

# (i) Monthly dividend rate by stock
gg_arate_dividend = (ggplot(dividend,aes(x='year',y='pct',group='ticker',color='tt')) + 
    theme_bw() + geom_line() + geom_point(size=0.5) + 
    scale_y_continuous(limits=[0,0.2],labels=percent_format()) + 
    labs(y='Annual dividend rate') + 
    ggtitle('sum(dividends)/average(open price)\nDividends extrapolated for incomplete years') + 
    facet_wrap('~q4',labeller=labeller(q4=di_qq)))
gg_save('gg_arate_dividend.png',dir_figures,gg_arate_dividend,8,6)

# (ii) Prospective vs retrospective dividend performance
cn_core = ['ticker','date','price','dividend']
perf_div = df_reit[cn_core].copy()#.query('ticker=="AP-UN.TO" | ticker == "BPY-UN.TO"')
l_seq = np.arange(-12,12+1)
l_seq = pd.Series(np.setdiff1d(l_seq,0))
tmp = pd.concat([perf_div.groupby('ticker').dividend.shift(l) for l in l_seq],1)
tmp.columns = l_seq #pd.Series(np.where(l_seq<0,'lead','lag')) + '_' + l_seq.abs().astype(str)
perf_div = pd.concat([perf_div,tmp],1)
pref_div = perf_div[perf_div[l_seq[l_seq>0]].notnull().all(1)]
perf_div = perf_div.melt(cn_core,None,'ll').assign(ll=lambda x: x.ll.astype(int))
perf_div = perf_div.sort_values(['ticker','date','ll']).reset_index(None,True)
perf_div = perf_div.assign(offset=lambda x: np.where(x.ll < 0, 'lead', 'lag'))
# Technically dropping the dividend of that contemporaneous month
perf_div = perf_div.drop(columns='dividend').rename(columns={'value':'dividend'})
perf_div = perf_div.groupby(['ticker','date','price','offset']).apply(lambda x: 
    pd.Series({'dividend':x.dividend.mean(), 'n':x.dividend.notnull().sum()}))
perf_div = perf_div.reset_index().assign(n=lambda x: x.n.astype(int))
perf_div = perf_div.assign(dividend=lambda x: x.dividend*12).query('n>0')
# Compare
perf_div = perf_div.assign(arate=lambda x: x.dividend/x.price)
offset_dmax = perf_div.query('offset=="lead" & n == 12').date.max()
ord_ticker = perf_div.groupby('ticker').arate.mean().sort_values(ascending=False).index.values
perf_div.ticker = pd.Categorical(perf_div.ticker,ord_ticker)
# perf_div = perf_div.merge(dat_reit,'left','ticker')

gg_perf_div = (ggplot(perf_div,aes(x='date',y='arate',color='offset')) + 
    theme_bw() + geom_point(size=0.5) + geom_line() + 
    facet_wrap('~ticker',nrow=4,scales='free') + 
    scale_y_continuous(labels=percent_format()) + 
    labs(y='Yearly rate of return',x='Date') + 
    geom_vline(xintercept=offset_dmax) + 
    ggtitle('Vertical line show extrapolation point') + 
    scale_color_discrete(name='Perspective',labels=['Retrospective','Prospective']) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y') + 
    theme(axis_text=element_blank(),axis_ticks=element_blank(),
          subplots_adjust={'hspace': 0.15,'wspace': 0.05}))
gg_save('gg_perf_div.png',dir_figures,gg_perf_div,20,8)

# Difference
err_div = perf_div.pivot_table('arate',['ticker','date'],'offset').reset_index().dropna()
err_div = err_div.assign(extrap=lambda x: np.where(x.date>offset_dmax,True,False))
err_div = err_div.merge(dat_reit[['ticker','tt']])
err_div = err_div.assign(err = lambda x: (x.lead - x.lag).abs())
err_div_yy = add_date_int(err_div).groupby(['tt','extrap','year']).err.describe()
# err_div_yy.columns = ['med','lb','ub']
err_div_yy = err_div_yy[['mean','std']].reset_index()

gg_err_div = (ggplot(err_div_yy.query('extrap==False'),aes(x='year',y='mean',color='tt')) + 
    theme_bw() + geom_point() + geom_line() + 
    geom_linerange(aes(ymin='mean-std',ymax='mean+std')) + 
    labs(y='Percent w/ 1std',x='Date') + 
    theme(subplots_adjust={'wspace':0.1}) + 
    facet_wrap('~tt') + 
    scale_y_continuous(labels=percent_format()) + 
    ggtitle('Average absolute error prospective vs retrospective'))
gg_save('gg_err_div.png',dir_figures,gg_err_div,10,3)

# Find the most consistent
best_err = err_div.groupby(['ticker','extrap']).err.max().reset_index().query('extrap==False')
best_err = best_err.sort_values('err').merge(dat_reit)
ethresh = 0.03
ticker_best_err = best_err.query('err < @ethresh').ticker.to_list()
tmp = perf_div[perf_div.ticker.isin(ticker_best_err)]

gg_best_err = (ggplot(tmp,aes(x='date',y='arate',color='offset')) + 
    theme_bw() + geom_point(size=0.5) + geom_line() + 
    facet_wrap('~ticker',nrow=2) + 
    scale_y_continuous(labels=percent_format()) + 
    labs(y='Yearly rate of return',x='Date') + 
    geom_vline(xintercept=offset_dmax) + 
    ggtitle('Vertical line show extrapolation point') + 
    scale_color_discrete(name='Perspective',labels=['Retrospective','Prospective']) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y') + 
    theme(subplots_adjust={'hspace': 0.15,'wspace': 0.05}))
gg_save('gg_best_err.png',dir_figures,gg_best_err,10,6)

########################
# -- (4) REIT INDEX -- #

# Calculate with cap gains/dividends separated
di_msr = {'price':'Price','dd':'Dividends','dprice':'Price +  Dividends'}
tmp1 = make_index(df_reit, 'ticker', add_cap=True, add_div=True).assign(msr='dprice')
tmp2 = make_index(df_reit, 'ticker', add_cap=True, add_div=False).assign(msr='price')
tmp3 = make_index(df_reit, 'ticker', add_cap=False, add_div=True).assign(msr='dd')
reit_index = pd.concat([tmp1, tmp2, tmp3]).reset_index(None,True)
reit_index = reit_index.assign(msr=lambda x: pd.Categorical(x.msr.map(di_msr),list(di_msr.values())))
tmp = pd.DataFrame({'date':pd.to_datetime('2017-01-01'),'y':110,'label':'TSX'},index=[0])

gg_reit_idx = (ggplot(reit_index,aes(x='date',y='budget',color='msr')) + 
    theme_bw() + geom_line() + labs(y='Index value') + 
    geom_line(aes(x='date',y='tsx'),color='black',data=df_cpi_tsx) + 
    geom_text(aes(x='date',y='y',label='label'),data=tmp,color='black') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m'))
gg_save('gg_reit_idx.png',dir_figures,gg_reit_idx,7,4.5)

# Repeat with bootstraps
n_bs = 1000
stime = time()
holder = []
for i in range(n_bs):
    n_left, n_i = n_bs - (i+1), (i+1)
    tmp1 = make_index(df_reit, 'ticker', add_cap=True, add_div=True, boot=True, seed=i).assign(msr='dprice')
    tmp2 = make_index(df_reit, 'ticker', add_cap=True, add_div=False, boot=True, seed=i).assign(msr='price')
    tmp3 = make_index(df_reit, 'ticker', add_cap=False, add_div=True, boot=True, seed=i).assign(msr='dd')
    if n_i % 5 == 0:
        dtime = time() - stime
        rate = n_i / dtime
        seta = n_left / rate
        print('Iteration %i, ETA: %i seconds' % (n_i, seta)) 
    tmp4 = pd.concat([tmp1,tmp2,tmp3]).drop(columns=['cgains','dividend'])
    holder.append(tmp4)
# Merge and save for later
reit_bs = pd.concat(holder).reset_index(None,True)
reit_bs.to_csv('reit_bs.csv',index=False)
# Get 95% CI
alpha = 0.05
reit_bs_qq = reit_bs.groupby(['msr','date']).budget.quantile([alpha/2,1-alpha/2]).reset_index().pivot_table('budget',['msr','date'],'level_2')
reit_bs_qq.columns = ['lb','ub']
reit_bs_qq = reit_bs_qq.reset_index().assign(msr=lambda x: x.msr.map(di_msr))
reit_bs_qq = reit_index.merge(reit_bs_qq)

gg_reit_idx_bs = (ggplot(reit_bs_qq,aes(x='date',y='budget',color='msr')) + 
    theme_bw() + geom_line() + labs(y='Index value') + 
    facet_wrap('~msr',nrow=1) + 
    geom_ribbon(aes(ymin='lb',ymax='ub',fill='msr'),alpha=0.5) + 
    guides(color=False,fill=False) + 
    geom_line(aes(x='date',y='tsx'),color='black',data=df_cpi_tsx) + 
    geom_text(aes(x='date',y='y',label='label'),data=tmp,color='black') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=45)) + 
    scale_x_datetime(date_breaks='5 years',date_labels='%Y-%m'))
gg_save('gg_reit_idx_bs.png',dir_figures,gg_reit_idx_bs,13,4.5)

