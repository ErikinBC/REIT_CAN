# Load libraries
import os
import pickle
import pandas as pd
import numpy as np
from plotnine import *
from plotnine.labels import ggtitle
from scipy.stats import norm
import holidays
from funs_stats import get_delta, quadrant_pr, get_CI
from funs_support import ym2date, gg_save, idx_first

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

# Common variables
dmin = '2010-01-01'
dfmt = '%Y-%m-%d'
idx = pd.IndexSlice

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

# (i) Calculate for stock
mm_stock = pd.concat([df_other[cn_ticker],df_reit[cn_ticker]])
mm_stock = get_delta(mm_stock,'price','ticker', 1).dropna()
mm_stock = mm_stock.assign(sidx=lambda x: np.sign(x.mm).astype(int))
mm_stock = mm_stock.drop(columns=['mm','price'])#.rename(columns={'price':'idx'})
ticker12 = mm_stock[mm_stock.date>=dmin].ticker.value_counts().reset_index().query('ticker>=12')['index']
mm_stock = mm_stock[mm_stock.ticker.isin(ticker12)].reset_index(None, True)
# (ii) Get "any" negative change CREA/Teranet/Housing type
mm_hpi_s = df_hpi.drop(columns=['year','idx']).pivot_table('sidx',['date','city'],['tt','hpi'])
mm_hpi = (mm_hpi_s==-1).any(1).replace(True,-1).replace(False,1).reset_index().rename(columns={0:'sidx'})
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

can_holidays = holidays.CAN()
on_holidays = holidays.CountryHoliday('CAN', prov='ON')

# --- (i) release dates --- #
# Load the release dates
release = pd.read_csv('release_date.csv')
release = pd.to_datetime(release.date)
dat_release = pd.DataFrame({'date':release,'num':release.dt.day, 'day':release.dt.day_name()})
dat_release.sort_values(['num','day']).reset_index(None,True)
# (i) if 15th is a weekday, then it's released on that day
# (ii) if the 15th is a Saturday, then it's released on the Friday (14th) or the Monday (17th)
# (iii) if the 15th is a Sunday, then it's released on the Monday

dseq = pd.date_range(dmin, df_hpi.date.max()+pd.DateOffset(months=1), freq='1M')
dseq = pd.to_datetime(pd.Series(dseq.strftime('%Y-%m')+'-15'))
df_dseq = pd.DataFrame({'date':dseq, 'day':dseq.dt.day_name()})
df_dseq = df_dseq.assign(date2=lambda x: np.where(x.day=='Saturday',x.date+pd.DateOffset(days=2),
    np.where(x.day == 'Sunday',x.date+pd.DateOffset(days=1), x.date) ))
# Check for holidays
tmp1 = pd.Series([can_holidays.get(d.strftime(dfmt)) for d in df_dseq.date2])
tmp2 = pd.Series([on_holidays.get(d.strftime(dfmt)) for d in df_dseq.date2])
assert np.all(tmp1.isnull() == tmp2.isnull())
df_dseq = df_dseq.assign(date3=lambda x: np.where(tmp1.notnull(),
    np.where(x.date2.dt.day==17, x.date2-pd.DateOffset(days=3), x.date2+pd.DateOffset(days=1)),
    x.date2))
df_dseq = df_dseq.rename(columns={'date3':'release'}).assign(day=lambda x: x.date2.dt.day_name(), num=lambda x: x.date2.dt.day)
df_dseq.drop(columns=['date','date2'], inplace=True)
df_dseq = df_dseq.assign(date=lambda x: pd.to_datetime(x.release.dt.strftime('%Y-%m')+'-01'))

# --- (ii) # of negative indicators --- #
hpi_sign = mm_hpi_s.where(mm_hpi_s==-1, 0).abs()
hpi_sign = hpi_sign.iloc[hpi_sign.index.sortlevel('city')[1]]
hpi_sign = hpi_sign[hpi_sign.index.get_level_values('date') >= dmin]
hpi_sign = hpi_sign[hpi_sign.index.get_level_values('city').isin(cities)]
hpi_sign = hpi_sign.sum(1).reset_index().rename(columns={0:'nneg'})
hpi_sign = hpi_sign.merge(df_dseq[['release','date']],'left','date')

# Merge the stock data
stock_daily = pd.concat([df_reit_daily, df_other_daily]).reset_index(None, True)

holder = []
for city in cities:
    print('--- city: %s ---' % (city))
    nneg_dates = hpi_sign.query('city == @city & nneg >= 1').release.reset_index(None, True)
    for date in nneg_dates:
        # print('date: %s' % date)
        date_max = date + pd.DateOffset(months=1)
        tmp_df = stock_daily.query('date >= @date & date < @date_max').reset_index(None, True)
        tmp_price = tmp_df.groupby('ticker').apply(lambda x: x.head(1).price).reset_index().drop(columns='level_1')
        tmp_price = tmp_price.assign(qty=lambda x: 100/x.price).drop(columns='price')
        tmp_df = idx_first(tmp_df, 'ticker', 'date', 'price').assign(short=lambda x: 100-x.price)
        tmp_df = tmp_df.drop(columns='price').merge(tmp_price)
        tmp_df = tmp_df.assign(profit=lambda x: x.short - x.dividend*x.qty).drop(columns=['qty','dividend'])
        tmp_df = tmp_df.assign(ndays=lambda x: (x.date - date).dt.days, date=date, city=city)
        holder.append(tmp_df)

# --- (iii) Analyze shorting performance --- #
res_short = pd.concat(holder).reset_index(None, True)
cn_short = ['city','ticker','ndays']
res_short_lbub = res_short.groupby(cn_short).profit.apply(lambda x: 
    pd.Series({'mi':x.min(), 'mx':x.max(), 'mu':x.mean()})).reset_index()
res_short_lbub = res_short_lbub.pivot_table('profit',cn_short,'level_'+str(len(cn_short))).reset_index()
res_short_lbub = res_short_lbub.assign(city=lambda x: pd.Categorical(x.city, cities))
# res_short_lbub['is_reit'] = res_short_lbub.ticker.isin(df_reit.ticker.unique())

# Plot it
for city in cities:
    print('--- city: %s ---' % (city))
    tmp_fn = 'gg_short_' + city + '.png'
    tmp_gtit = city + ' - $100 short sale (dots show mean)'
    tmp_df = res_short_lbub.query('city == @city')
    order_ticker = list(tmp_df.groupby('ticker').mu.mean().sort_values(ascending=False).index)
    tmp_df = tmp_df.assign(ticker = lambda x: pd.Categorical(x.ticker, order_ticker))
    gg_tmp = (ggplot(tmp_df, aes(x='ndays',y='mu',color='city',fill='city')) + 
        theme_bw() + geom_line() + geom_point() + 
        guides(color=False, fill=False) + ggtitle(tmp_gtit) + 
        facet_wrap('~ticker',scales='free_y') + 
        geom_ribbon(aes(ymin='mi',ymax='mx'),alpha=0.05) + 
        geom_hline(yintercept=0,linetype='--') + 
        theme(subplots_adjust={'wspace': 0.25}) + 
        labs(y='Profit/Loss',x='# of days since short'))
    gg_save(tmp_fn, dir_figures, gg_tmp, 16, 10)

#####################################
# --- (4) RANKING WITH ANALYSIS --- #

# Mean of mean's?
# Max vs min?
# Combinations?




