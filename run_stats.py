# Load libraries
import os
import pickle
import pandas as pd
import numpy as np
from plotnine import *
from scipy.stats import norm
from statsmodels.stats.proportion import proportion_confint as prop_CI

from funs_support import ym2date, gg_save


# Load the data
dir_base = os.getcwd()
dir_figures = os.path.join(dir_base, 'figures')

with open('data_for_plots.pickle', 'rb') as handle:
    di_storage = pickle.load(handle)

df_tera = di_storage['teranet']
tera_w = ym2date(di_storage['tera_w'])
df_crea = di_storage['crea']
df_cpi_tsx = di_storage['cpi_tsx']
df_lf = di_storage['lf']
df_mort_tera = di_storage['mort']
df_reit = di_storage['reit']
df_reit_daily = di_storage['reit_daily']
df_shiller = di_storage['shiller']
df_other = di_storage['other']
df_other_daily = di_storage['other_daily']

cities = ['Canada','Toronto','Vancouver']
cn_tera = ['date','city','mm']
cn_ticker = ['ticker','date','price']
cn_gg = ['hpi','city','tt']
di_hpi = {'crea':'CREA', 'tera':'Teranet'}
di_tt = {'aggregate':'Aggregate', 'apt':'Apartment', 'row':'Townhouse', 'sfd':'Detached'}


# Get the period change
def get_delta(df, cn, gg, k=1):
    return df.assign(mm=lambda x: x[cn]/x.groupby(gg)[cn].shift(k)-1)

def quadrant_pr(df, cn_y, cn_x):
    df = df.copy()
    cn = [cn_y, cn_x]
    df[cn] = df[cn].apply(lambda x: np.sign(x).astype(int), 0)
    res = df.groupby(cn).size().reset_index().rename(columns={0:'n'})
    # TPR/TNR
    ntp = res[(res[cn_y] == 1) & (res[cn_x] == 1)].n.sum()
    nump = res[res[cn_y] == 1].n.sum()
    ntn = res[(res[cn_y] == -1) & (res[cn_x] == -1)].n.sum()
    nn = res[res[cn_y] == -1].n.sum()
    tpr, tnr = ntp/nump, ntn/nn
    # PPV/NPV
    npp = res[res[cn_x] == 1].n.sum()
    npn = res[res[cn_x] == -1].n.sum()
    ppv = ntp / npp
    npv = ntn / npn
    res = pd.DataFrame({'metric':['tpr','tnr','ppv','npv'],
                  'value':[tpr, tnr, ppv, npv], 'n':[nump, nn, npp, npn]})
    return res

# For a dataframe with the percentage and n
def get_CI(df, cn_p, cn_n, method='beta'):
    return df.assign(lb=lambda x: prop_CI(x[cn_p]*x[cn_n],x[cn_n],method=method)[0],
                    ub=lambda x: prop_CI(x[cn_p]*x[cn_n],x[cn_n],method=method)[1])

##########################################
# --- (1) APPLY SEASONAL ADJUSTMENTS --- #

from statsmodels.tsa.seasonal import seasonal_decompose

# Merge CREA and Tera
df_hpi = pd.concat([df_crea.assign(hpi='crea'),
            df_tera.assign(tt='aggregate')[df_crea.columns].assign(hpi='tera')])
df_hpi.reset_index(None,True,True)
tmp = df_hpi.groupby(cn_gg).apply(lambda x: pd.DataFrame({
    'trend':seasonal_decompose(x=x.idx.values, model='additive', period=12, two_sided=True).trend}))
tmp = tmp.reset_index().rename(columns={'level_3':'ridx'})
df_hpi = df_hpi.assign(ridx=df_hpi.groupby(cn_gg).cumcount()).merge(tmp).dropna()
df_hpi = df_hpi.drop(columns=['ridx','idx']).rename(columns={'trend':'idx'})
df_hpi = df_hpi.assign(sidx=lambda x: x.groupby(cn_gg).idx.diff(1)).dropna().reset_index(None,True)
df_hpi = df_hpi.assign(year=lambda x: x.date.dt.year, sidx=lambda x: np.sign(x.sidx).astype(int))

df_hpi_sidx = df_hpi.pivot_table(index=cn_gg+['year'],columns='sidx',values='idx',aggfunc='count')
df_hpi_sidx = df_hpi_sidx.fillna(0).astype(int).reset_index().melt(cn_gg+['year'],None,None,'n')
df_hpi_sidx = df_hpi_sidx.assign(hpi=lambda x: x.hpi.map(di_hpi),tt=lambda x: x.tt.map(di_tt))
# Plot the number of negative months
tmp = df_hpi_sidx.query('sidx == -1 & city.isin(@cities)', engine='python').drop(columns='sidx')
gg_hpi_sidx = (ggplot(tmp, aes(x='year',y='n',color='tt')) + 
    geom_point() + geom_line() + theme_bw() + 
    scale_y_continuous(breaks=list(range(1,13,1))) + 
    scale_x_continuous(breaks=list(range(2005,2021,1))) + 
    facet_grid('city~hpi') + 
    theme(axis_title_x=element_blank(),axis_text_x=element_text(angle=90)) + 
    scale_color_discrete(name='Housing type') + 
    ggtitle('Index adjusted for seasonality') + 
    labs(x='Year',y='# of negative months'))
gg_save('gg_hpi_sidx.png', dir_figures, gg_hpi_sidx, 8, 8)


###########################################
# --- (2) QUADRANT STRATEGY (MONTHLY) --- #

dmin = '2010-01-01'

# (i) Calculate for stock
mm_stock = pd.concat([df_other[cn_ticker],df_reit[cn_ticker]])
mm_stock = get_delta(mm_stock,'price','ticker', 1).dropna()
mm_stock = mm_stock.assign(sidx=lambda x: np.sign(x.mm).astype(int))
mm_stock = mm_stock.drop(columns=['mm','price'])#.rename(columns={'price':'idx'})
ticker12 = mm_stock[mm_stock.date>=dmin].ticker.value_counts().reset_index().query('ticker>=12')['index']
mm_stock = mm_stock[mm_stock.ticker.isin(ticker12)].reset_index(None, True)
# (ii) Get "any" negative change CREA/Teranet/Housing type
mm_hpi = df_hpi.drop(columns=['year','idx']).pivot_table('sidx',['date','city'],['tt','hpi'])
mm_hpi = (mm_hpi==-1).any(1).replace(True,-1).replace(False,1).reset_index().rename(columns={0:'sidx'})
mm_hpi = mm_hpi.merge(mm_stock,'inner','date',suffixes=('_hpi','_stock')) #.drop(columns=['year','idx'])
mm_hpi = mm_hpi.sort_values(['city','ticker','date'])
# Subset to 2010 onwards with n>12 and key cities
mm_hpi = mm_hpi.query('city.isin(@cities) & date>=@dmin',engine='python').reset_index(None,True)

# (iii) Two-columns
tmp_cn = ['city','ticker']
pr_hpi = mm_hpi.groupby(tmp_cn).apply(lambda x: quadrant_pr(x, cn_y='sidx_stock', cn_x='sidx_hpi') )
pr_hpi = pr_hpi.reset_index().drop(columns='level_'+str(len(tmp_cn)))
pr_hpi = pr_hpi.assign(metric=lambda x: x.metric.str.upper())#, tt=lambda x: x.tt.map(di_tt)
pr_hpi = get_CI(pr_hpi, 'value', 'n', 'beta')
pr_hpi = pr_hpi.assign(se=lambda x: np.sqrt(x.value*(1-x.value)/x.n), 
                      ticker2=lambda x: x.ticker.str.replace('\\-UN|\\.TO',''))
pr_hpi['is_reit'] = pr_hpi.ticker.isin(df_reit.ticker.unique())

cn_tn = ['ticker','name']
di_ticker = pr_hpi.groupby(['ticker','ticker2']).size().reset_index().drop(columns=[0])
di_ticker = di_ticker.merge(pd.concat([df_other[cn_tn], df_reit[cn_tn]]).groupby(cn_tn).size().reset_index().drop(columns=[0]))

# Loop over each metric
for met in ['TNR','NPV']:
    gtit = 'Metric = %s' % met
    print(gtit)
    tmp = pr_hpi.query('metric==@met').drop(columns=['metric']).dropna().reset_index(None, True)
    # Get rough order
    order_ticker = list(tmp.groupby('ticker2').value.mean().sort_values().index)
    tmp.ticker2 = pd.Categorical(tmp.ticker2,order_ticker)
    # Calculate the "best" within each category
    tmp2 = tmp.loc[tmp.groupby(['city']).apply(lambda x: x.value.idxmax()).values]
    tmp2 = tmp2[['city','value','se']].rename(columns={'value':'value_max', 'se':'se_max'})
    tmp = tmp.merge(tmp2,'left')
    tmp = tmp.assign(pv=lambda x: norm.cdf(-(x.value_max-x.value)/np.sqrt(x.se**2 + x.se_max**2)))

    # Plot it
    fn = 'gg_quadrant_'+met+'.png'
    gtit2 =gtit+'\nShaded out regions are statistically worse than best'
    gg_tmp = (ggplot(tmp, aes(x='ticker2',y='value',color='is_reit',alpha='pv<0.05')) + 
        theme_bw() + ggtitle(gtit2) + 
        labs(y='Percentage') +  coord_flip() + 
        facet_wrap('~city') + geom_point() + 
        geom_linerange(aes(ymin='lb',ymax='ub')) + guides(alpha=False) + 
        scale_color_discrete(name='Is Reit?',labels=['No','Yes']) + 
        scale_alpha_manual(values=[1, 0.25]) + 
        theme(axis_title_y=element_blank(),legend_position='right') )
    gg_save(fn, dir_figures, gg_tmp, 10, 7)
    
cn_best = ['BTB','SOT','HOT','CWX']
di_ticker.query('ticker2.isin(@cn_best)',engine='python')


#########################################
# --- (3) SHORTING STRATEGY (DAILY) --- #

# When has the CREA/Teranet gone negative?
mm_hpi.assign(ss=lambda x: np.sign(x.value).astype(int)).groupby(['city','tt','ss']).size()



# SHORTING STRATEGY: SHORT STOCK WHEN THE CREA-HPI BECOMES AVAILABLE AND GOES NEGATIVE
# DISTRIBUTION OF LOSSES AS A FUNCTION OF HOW LONG TO HOLD ONTO UNTIL FIRST POSITIVE TURN?
#   MAY NEED TO LOAD DAILY DATA AND ESTIMATE CREA RELEASE DAY


