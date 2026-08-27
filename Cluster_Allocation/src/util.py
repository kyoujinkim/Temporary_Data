import numpy as np
import pandas as pd
import cvxpy as cp

def qp_project_with_bounds(w, lb=0.05, ub_default=1.0, cap_assets=("Gold", "WTI"), cap=0.20):
    assets = w.index.tolist()
    n = len(assets)

    lower = pd.Series(lb, index=assets, dtype=float)
    upper = pd.Series(ub_default, index=assets, dtype=float)
    for a in cap_assets:
        if a in upper.index:
            upper[a] = cap

    # Define optimization variables
    w_target = w.values
    w_opt = cp.Variable(n)

    # Objective: Minimize the squared distance to the original weights (smooth transition)
    objective = cp.Minimize(cp.sum_squares(w_opt - w_target))

    # Constraints
    constraints = [
        cp.sum(w_opt) == 1.0,  # Fully constrained sum to 1
        w_opt >= lower.values,  # Lower bounds
        w_opt <= upper.values  # Upper bounds
    ]

    # Solve
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)  # ECOS is fast and built-in

    return pd.Series(w_opt.value, index=assets)

def project_with_bounds(w, lb=0.05, ub_default=1.0, cap_assets=("Gold", "WTI"), cap=0.20):
    w = w.copy()
    assets = w.index.tolist()

    lower = pd.Series(lb, index=assets, dtype=float)
    upper = pd.Series(ub_default, index=assets, dtype=float)
    for a in cap_assets:
        if a in upper.index:
            upper[a] = cap

    # 1) clip
    w = w.clip(lower=lower, upper=upper)

    # 2) 합계를 1로 재조정 (간단 방식)
    for _ in range(100):
        diff = 1.0 - w.sum()
        if abs(diff) < 1e-10:
            break
        free = (w > lower + 1e-12) & (w < upper - 1e-12)
        if free.sum() == 0:
            break
        w.loc[free] += diff / free.sum()
        w = w.clip(lower=lower, upper=upper)

    return w

def smooth_project_with_bounds(w, lb=0.05, ub_default=1.0, cap_assets=("Gold", "WTI"), cap=0.20, smoothness=1.0):
    w = w.copy()
    assets = w.index.tolist()

    lower = pd.Series(lb, index=assets, dtype=float)
    upper = pd.Series(ub_default, index=assets, dtype=float)
    for a in cap_assets:
        if a in upper.index:
            upper[a] = cap

    # 1) Soft-mapping using a Sigmoid function to strictly stay within bounds smoothly
    # 'smoothness' controls how strictly it mimics a hard clip. Lower = smoother.
    scaled_w = smoothness * w
    logistic_w = 1 / (1 + np.exp(-scaled_w))

    # Map from (0, 1) to (lower, upper)
    w_continuous = lower + (upper - lower) * logistic_w

    # 2) Smooth Normalization (Softmax-like adjustment so sum = 1)
    # Instead of hard-fixing free assets, we iteratively scale relative to available distance
    for _ in range(10):
        current_sum = w_continuous.sum()
        if abs(1.0 - current_sum) < 1e-6:
            break

        # Adjust proportional to how much room is left to the boundary
        if current_sum < 1.0:
            room = upper - w_continuous
            w_continuous += room * ((1.0 - current_sum) / room.sum())
        else:
            room = w_continuous - lower
            w_continuous -= room * ((current_sum - 1.0) / room.sum())

    return w_continuous