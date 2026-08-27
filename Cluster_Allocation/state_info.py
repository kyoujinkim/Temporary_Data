import numpy as np
import pandas as pd

from Cluster_Allocation.src.util import qp_project_with_bounds, smooth_project_with_bounds

assets = ['asset', 'market', 'bond']

for a in assets:

    idx = pd.read_excel(f"data/data.xlsx", sheet_name=f"data_{a}", index_col=0, header=1, parse_dates=True)
    idx_name = pd.read_excel(f"data/data.xlsx", sheet_name=f"name_{a}", header=0)
    idx_rtn = idx.pct_change(periods=1).dropna(how='all', axis=0)

    state = pd.read_csv('result/state_data.csv', index_col=0, parse_dates=True)

    idx_rtn = idx_rtn.reindex(state.index, method='ffill')
    idx_rtn['state'] = state['state']

    def remove_outliers_iqr(group):
        Q1 = group.quantile(0.15)
        Q3 = group.quantile(0.85)
        IQR = Q3 - Q1
        return group[~((group < (Q1 - 1 * IQR)) | (group > (Q3 + 1 * IQR))).any(axis=1)]

    idx_rtn_mean = idx_rtn.groupby('state').apply(remove_outliers_iqr).groupby(level=0).mean().drop('state', axis=1)
    idx_rtn_mean.columns = idx_name['Name'].values

    tmp_col = idx_rtn_mean.columns.to_list() + ['stat']
    idx_rtn.columns = tmp_col
    idx_ind = idx_rtn_mean
    idx_ind = idx_ind.apply(lambda x: x - x.min(), axis=1)
    idx_ind_w = idx_ind.div(idx_ind.sum(axis=1), axis=0)

    for i in idx_ind_w.index:
        w = qp_project_with_bounds(idx_ind_w.loc[i], lb=0.00, ub_default=1.0, cap_assets=['Gold', 'WTI'], cap=0.25)
        idx_ind_w.loc[i] = w

    idx_ind_w.to_csv(f'model/state_optimal_weight_{a}.csv', encoding='UTF-8-sig')

    '''def w_ind_by_pct(x):
        total = []
        rtn = idx_rtn.loc[x.name][idx_ind.columns]
        for c in cluster['state'].unique():
            w = idx_ind.loc[c].copy()
            w *= (~rtn.isna())
            w /= w.sum()
            w *= x[c]
            if c == x.cluster:
                w *= 1.5
            total.append(w)
        df = pd.concat(total, axis=1)
        return df.sum(axis=1) / df.sum(axis=1).sum()

    w_ind = X.apply(w_ind_by_pct, axis=1)
    a = 1'''