import pandas as pd

assets = ['asset', 'market', 'bond']

for a in assets:
    state = pd.read_csv('result/state_data.csv', index_col=0, parse_dates=True)
    opt_w = pd.read_csv(f'model/state_optimal_weight_{a}.csv', index_col=0)
    state_list = state.state.unique()

    def w_ind_by_pct(x):
        total = []
        for c in state_list:
            w = opt_w.loc[c].copy()
            w *= x[f'state_{c}']
            total.append(w)
        df = pd.concat(total, axis=1)
        return df.sum(axis=1) / df.sum(axis=1).sum()

    w_ind = state.apply(w_ind_by_pct, axis=1)

    w_ind.to_csv(f'result/weight_{a}.csv', encoding='UTF-8-sig')