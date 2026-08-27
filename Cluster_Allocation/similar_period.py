import numpy as np
import pandas as pd
import numpy as np
from tqdm import tqdm

state_df = pd.read_csv('result/state_data.csv', index_col=0, parse_dates=True)

# calculate and get similar state
# 2) state 확률 컬럼 자동 추출
# 예: state_0, state_1 ... 또는 숫자 컬럼
cand_cols = [c for c in state_df.columns if "state_" in str(c).lower()]
if len(cand_cols) == 0:
    # fallback: 숫자형 컬럼만 사용
    cand_cols = state_df.select_dtypes(include=[np.number]).columns.tolist()

P = state_df[cand_cols].dropna().copy()

# 혹시 각 행 합이 1이 아니면 정규화
row_sum = P.sum(axis=1).replace(0, np.nan)
P = P.div(row_sum, axis=0).dropna()

# 3) 현재 시점 벡터
latest = P.iloc[-13:].values  # shape: (K,)
l_norm = np.linalg.norm(latest, ord='fro')

# 4) 과거 각 시점과의 유사도 계산 (코사인 유사도)
# patch 단위로 나누기
total_patch = []
for i in tqdm(range(len(P)-6, 13, -6)):
    X = P.iloc[i-13:i].values
    x_norm = np.linalg.norm(X, ord='fro')

    # Element-wise multiplication and sum (Matrix dot product)
    matrix_dot = np.sum(X * latest)

    sim = matrix_dot / (x_norm * l_norm + 1e-12)

    total_patch.append([P.index[i], sim])

patch_df = pd.DataFrame(total_patch, columns=['index', 'sim']).set_index('index')
patch_df = patch_df.sort_values(by=['sim'], ascending=False).iloc[:10]
matched_state_df = state_df.loc[patch_df.index].copy()

current_state_row = state_df.iloc[[-1]].copy()
matched_with_sim_df = matched_state_df.join(patch_df, how='left')
current_state_row['sim'] = np.nan
similar_state = pd.concat([matched_with_sim_df, current_state_row], axis=0)

current_state_row.to_csv('result/current_state.csv')
similar_state.to_csv('result/most_similar_state.csv')
