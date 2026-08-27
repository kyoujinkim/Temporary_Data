import pickle

import datetime as dt
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import softmax
from sklearn.mixture import GaussianMixture


class Model():
    def __init__(self, train=False):
        self.train = train
        self.set_model()

    def set_model(self):
        if not self.train:
            with open(f'model/model.pkl', 'rb') as f:
                model = pickle.load(f)
            self.model = model
        else:
            self.model = GaussianHMM(n_components=4, n_iter=200, tol=0.001, covariance_type='full', random_state=0, verbose=True)

    def standardize_data(self, X, period, skip_col, date="2099-12-31"):
        # set std window
        X_copy = X.loc[:date].copy()

        if period == -1:
            _period = len(X_copy)
        else:
            _period = period

        X_mean = X_copy.rolling(_period, min_periods=1).mean().bfill()
        X_std = X_copy.rolling(_period, min_periods=1).std().bfill()
        X_mean = pd.DataFrame(X_mean, index=X.index, columns=X.columns).ffill()
        X_std = pd.DataFrame(X_std, index=X.index, columns=X.columns).ffill()

        Xs = ((X - X_mean) / X_std).bfill()

        if skip_col:
            for col in skip_col:
                if col in X.columns:
                    Xs[col] = X[col]

        self.mean = X_mean
        self.std = X_std

        return Xs, X_mean, X_std

    def unstandardize_data(self, X):
        # unstandardize X
        X = X * self.X_std + self.X_mean
        return X

    def fit(self, X, window=-1, end_date=-1, skip_col:list=None, enddate="2099-12-31"):
        # Use sklearn Gaussian Mixture Model
        if end_date != -1:
            X = X[X.index <= end_date]
        X_std, _, _ = self.standardize_data(X, period=window, skip_col=skip_col, date=enddate)

        self.model.fit(X_std)
        if self.train:
            self.save_model()

        return True

    def state_mean(self, skip_col:list=None):
        Z_means = pd.DataFrame(self.model.means_, columns=self.mean.columns)
        X_means = self.model.means_ * self.std.iloc[-1].values + self.mean.iloc[-1].values
        X_means = pd.DataFrame(X_means, columns=self.mean.columns)
        if skip_col:
            for col in skip_col:
                X_means[col] = Z_means[col]

        return X_means

    @staticmethod
    def adjust_state(Z):
        X = pd.DataFrame(Z)
        # set flag which signal the cluster change
        flag_list = [0]
        flag = 0
        for i in range(1, len(X)):
            if flag == 0 and X.state[i] != X.state[i - 1]:
                flag = 1
            elif flag == 1:
                if X.state[i] == X.state[i - 1]:
                    flag = 0
                else:
                    flag = 0
            else:
                pass
            flag_list.append(flag)

        X['flag'] = flag_list
        # if flag == 1, change X.cluster and cluster_num as previous value
        X['state_nadj'] = X['state'].copy()
        X['state'] = X['state'].where(X['flag'] == 0, X['state'].shift(1))

        return X['state']

    def transmat(self):
        X_cluster = self.astate
        X_cluster_cons_dup = X_cluster.loc[X_cluster.shift(1) != X_cluster]
        # add period column to X_cluster_cons_dup
        X_period = X_cluster_cons_dup.index[1:].append(
            pd.Index([pd.to_datetime(dt.date.today())])) - X_cluster_cons_dup.index
        X_period = pd.Series(X_period.days.values, index=X_cluster_cons_dup.index, name='period')
        # concat period to X_cluster_cons_dup
        X_cluster_cons_dup = pd.concat([X_cluster_cons_dup, X_period], axis=1)
        # drop if period is less than 7 days
        X_cluster_cons_dup = X_cluster_cons_dup[X_cluster_cons_dup['period'] > 7]
        X_cluster_cons_dup = X_cluster_cons_dup.loc[X_cluster_cons_dup.state.shift(1) != X_cluster_cons_dup.state]
        transition = pd.crosstab(X_cluster_cons_dup.state.shift(1), X_cluster_cons_dup.state, normalize='index')
        cluster_period = X_cluster_cons_dup.groupby('state').agg(['mean', 'median', 'std'])

        if self.model_name == 'GMM':
            return transition, cluster_period

        elif self.model_name == 'HMM':
            transition = pd.DataFrame(self.model.transmat_)
            return transition, cluster_period

    def proba(self, X, temperature=1.0, skip_col:list=None, enddate="2099-12-31"):
        # Get the probability of each state
        X_std, _, _ = self.standardize_data(X, period=-1, skip_col=skip_col, date=enddate)

        proba = self.model._compute_log_likelihood(X_std)

        Z_proba = softmax(proba/temperature, axis=1)
        Z_proba = pd.DataFrame(Z_proba, index=X.index, columns=[f'state_{i}' for i in range(self.model.n_components)])
        return Z_proba

    def predict(self, X, skip_col:list=None, enddate="2099-12-31"):
        # Predict the state for each observation
        X_std, _, _ = self.standardize_data(X, period=-1, skip_col=skip_col, date=enddate)

        Z = self.model.predict(X_std)
        Z = pd.Series(Z, index=X.index, name='state')
        self.state = Z
        self.astate = self.adjust_state(Z)

        return self.astate

    def save_model(self):
        with open(f'model/model.pkl', 'wb') as f:
            pickle.dump(self.model, f)


if __name__ == '__main__':

    train = True
    X = pd.read_csv('data/db.csv', index_col=0, parse_dates=True)
    model = Model(train=train)

    period = 52*3
    enddate = '2025-07-31'
    skip_col = ['em_hsd', 'eu_hsd', 'us_hsd']

    if train:
        model.fit(X, window=period, end_date=enddate, skip_col=skip_col)

    state = model.predict(X, skip_col=skip_col, enddate=enddate)
    prob = model.proba(X, temperature=2, skip_col=skip_col)
    # 시계열 정보
    export_data = pd.concat([X, state, prob], axis=1).dropna()
    export_data.groupby('state').median().to_csv('model/state_info.csv')
    export_data.to_csv(f'result/state_data.csv', index=True)