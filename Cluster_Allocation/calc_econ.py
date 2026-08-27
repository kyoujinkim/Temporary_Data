import numpy as np
import pandas as pd

country_list = ['em', 'eu', 'us']

hsd = pd.DataFrame()
for c in country_list:
    econ = pd.read_csv(f'data/hs_delta_{c}.csv', index_col=0, parse_dates=True).rolling(window=5, center=False).mean()
    econ.columns = ['econ', 'o1', 'o6']
    hsd[f'{c}_hsd'] = econ[['econ']]

fx = pd.read_excel('data/econ.xlsx', sheet_name='fx', index_col=0, parse_dates=True)
fx.columns = ['usd','jpy','eur']
fx_rtn = fx.pct_change(periods=1).dropna(how='all', axis=0)
fx_rtn = (fx_rtn + 1).rolling(window=4).apply(np.prod, raw=True).dropna(how='all', axis=0) - 1

econ = pd.concat([hsd, fx_rtn], axis=1).ffill()
econ = econ.asfreq('W-FRI').dropna()

econ.to_csv('data/db.csv')