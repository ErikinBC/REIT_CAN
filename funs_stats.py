# Load libraries
import numpy as np
import pandas as pd

def boot_vec(x,boot):
    if boot:
        x = x.copy()
        x = x.sample(frac=1,replace=True)
    return x

# FUNCTION TO CALCULATE INDEX FOR A STOCK SERIES
def make_index(x, cn_wide, add_cap=True, add_div=True, boot=False, seed=1):
    # x = df_reit.drop(columns=['name','tt']).copy(); cn_wide='ticker'; add_div=True; add_cap=True; boot=True; seed=1
    assert isinstance(x,pd.DataFrame)
    cn_req = ['date','price','dividend']
    assert x.columns.isin(cn_req).sum() == len(cn_req)
    if boot:
        np.random.seed(seed)
    # Pivot wide the prices and dividends
    x_price = x.pivot('date',cn_wide,'price')
    x_div = x.pivot('date',cn_wide,'dividend')
    # Get order of missingness
    cn_omiss = list(x_price.isnull().sum().sort_values().index)
    x_price = x_price[cn_omiss]
    x_div = x_div[cn_omiss]
    n_t = len(x_price)
    # Number of active stocks
    n_active = x_price.notnull().sum(1)
    # Initial budget and shares allocation
    budget = 100
    price_old = boot_vec(x_price.iloc[0].dropna(), boot)
    shares_old = budget / price_old / n_active[0]
    holder = np.zeros([n_t, 3])
    holder[0] = [budget, 0, 0]
    for i in range(1, n_t):
        # (i) Sell shares in new period
        price_new = x_price.iloc[i][price_old.index]
        cgains = np.sum(price_new * shares_old) - budget
        # (ii) Collect dividends
        dividends = np.sum(x_div.iloc[i][price_old.index] * shares_old)
        # (iii) Update budget
        if add_cap:
            budget += cgains
        if add_div:
            budget += dividends
        # (iv) Buy new shares
        price_old = boot_vec(x_price.iloc[i].dropna(), boot)
        shares_old = budget / price_old / n_active[i]
        holder[i] = [budget, cgains, dividends]
    # Format dataframe then return
    res = pd.DataFrame(holder,columns=['budget','cgains','dividend'])
    res.insert(0,'date',x_price.index)
    return res