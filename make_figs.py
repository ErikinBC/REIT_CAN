import numpy as np
import pandas as pd
import os
from plotnine import *
from mizani.formatters import percent_format
import pickle

from plotnine.labels import ggtitle
from support_funs import makeifnot, gg_save, ym2date, gg_color_hue

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

assert np.issubdtype(df_cpi_tsx.date,np.datetime64)

###################
# -- (3) REITS -- #



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

df_hpi_vs_stats = df_tera.query('city=="Canada"')[['date','idx']].merge(df_cpi_tsx,'left')
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







