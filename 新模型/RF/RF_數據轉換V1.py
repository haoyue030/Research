'''
特徵組合、參數篩選（RandomForest + Bayesian Optimization）
'''
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import boxcox

from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---- 新增：Bayesian Optimization 套件 ----
from skopt import gp_minimize
from skopt.space import Categorical
from skopt.utils import use_named_args

# ----------------------------------------
# 1) 讀取並做必要的數據轉換
# ----------------------------------------
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

# Yeo–Johnson
for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

# 潮位平方
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2

# 波能 & 功率 log1p
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

# 尖峰週期 Box–Cox (λ=2.4217)
x = df['尖峰週期']
x_pos = x - x.min() + 1e-6
df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

# 目標
y_all = df['y'].values

# ----------------------------------------
# 2) 候選特徵
# ----------------------------------------
feature_columns = [
    '風速', '氣壓',
    'wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos',
    '波高_YJ','降雨_YJ','暴風半徑_YJ',
    '潮位_BC2',
    '波能_log1p','功率_log1p',
    '尖峰週期_BC'
]

# ----------------------------------------
# 3) 切分前標準化：這裡只示範 Robust
# ----------------------------------------
X_full_raw = df[feature_columns].values

scalers = {
    'Robust': RobustScaler().fit(X_full_raw)
}
X_full_scaled = {name: sc.transform(X_full_raw) for name, sc in scalers.items()}

feat_to_idx = {f: i for i, f in enumerate(feature_columns)}

# ----------------------------------------
# 4) 特徵子集大小範圍 & Bayesian 搜尋空間
# ----------------------------------------
min_features = 3
max_features = 13

# 原本是固定的列表，這裡用 Categorical 保留離散值的特性
space = [
    Categorical([100, 150, 200, 250], name='n_estimators'),
    Categorical([3, 5, 7],                name='max_depth'),
    Categorical([30, 40, 50,60],     name='rf_seed')
]

# 單一「特徵子集」的 Bayesian Optimization
def bo_search_single_subset(X_sub, y_all, n_calls=25, random_state_bo=0):
    """
    對某一個 X_sub（已選好特徵的子矩陣）做 Bayesian Optimization，
    調整 n_estimators, max_depth, rf_seed，目標是最大化 R2_test。
    """

    # 固定切分方式，和你原本一樣 random_state=0
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_all, test_size=0.3, random_state=0
    )

    @use_named_args(space)
    def objective(n_estimators, max_depth, rf_seed):
        # 注意：gp_minimize 是「最小化」，所以回傳 -R2_test
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rf_seed
        )
        model.fit(X_train, y_train)
        y_pred_te = model.predict(X_test)
        r2_te = r2_score(y_test, y_pred_te)
        return -r2_te

    # 執行 Bayesian Optimization
    res = gp_minimize(
        func=objective,
        dimensions=space,
        n_calls=n_calls,
        random_state=random_state_bo,
        n_initial_points=10  # 前幾點用 random 探索
    )

    # 取出最佳組合
    best_params = {dim.name: val for dim, val in zip(space, res.x)}
    n_estimators_best = best_params['n_estimators']
    max_depth_best    = best_params['max_depth']
    rf_seed_best      = best_params['rf_seed']

    # 用最佳參數重新訓練一次，計算各種指標
    best_model = RandomForestRegressor(
        n_estimators=n_estimators_best,
        max_depth=max_depth_best,
        random_state=rf_seed_best
    )
    best_model.fit(X_train, y_train)

    y_pred_tr = best_model.predict(X_train)
    y_pred_te = best_model.predict(X_test)

    r2_tr = r2_score(y_train, y_pred_tr)
    r2_te = r2_score(y_test,  y_pred_te)

    mse_te  = mean_squared_error(y_test, y_pred_te)
    rmse_te = float(np.sqrt(mse_te))
    mae_te  = mean_absolute_error(y_test, y_pred_te)

    result = {
        'n_estimators': n_estimators_best,
        'max_depth':    max_depth_best,
        'RF_seed':      rf_seed_best,
        'R2_train':     r2_tr,
        'R2_test':      r2_te,
        'MSE_test':     mse_te,
        'RMSE_test':    rmse_te,
        'MAE_test':     mae_te
    }
    return result

def run_search_for_scaler(scaler_name, X_scaled_matrix, y_all):
    results = []

    for r in range(min_features, max_features + 1):
        for subset in combinations(feature_columns, r):
            # 方向特徵成對
            if ('wind_dir_sin' in subset) ^ ('wind_dir_cos' in subset):
                continue
            if ('wave_dir_sin' in subset) ^ ('wave_dir_cos' in subset):
                continue

            cols_idx = [feat_to_idx[f] for f in subset]
            X_sub = X_scaled_matrix[:, cols_idx]

            # 針對這組特徵，用 Bayesian Optimization 找最佳 RF 參數
            bo_result = bo_search_single_subset(X_sub, y_all)

            # 把 scaler / 特徵名稱補上
            bo_result.update({
                'Scaler':   scaler_name,
                'Features': subset
            })
            results.append(bo_result)

    return results

# ----------------------------------------
# 5) 針對 Scaler 分別搜尋並合併結果
# ----------------------------------------
all_results = []
for name in scalers.keys():
    res = run_search_for_scaler(name, X_full_scaled[name], y_all)
    all_results.extend(res)

# ----------------------------------------
# 6) 篩選 & 顯示：Top 10 by Train R²（且 Test R² > 門檻）
# ----------------------------------------
res_df = pd.DataFrame(all_results)

test_thr = 0.3   # Test R² 下限
top_k   = 10     # 取前幾名

if res_df.empty:
    print("※ 沒有任何結果，可能搜尋空間太小或哪裡出問題。")
else:
    top_df = (
        res_df.loc[res_df['R2_test'] > test_thr].copy()
              .sort_values(['R2_train', 'R2_test', 'RMSE_test'],
                           ascending=[False,     False,      True])
              .head(top_k)
    )

    # 美化 Features 顯示成 ('a','b','c')
    top_df['Features'] = top_df['Features'].apply(
        lambda t: '(' + ', '.join(f"'{f}'" for f in t) + ')'
    )

    # max_depth 是 int，直接轉成字串避免後續顯示問題
    top_df['max_depth'] = top_df['max_depth'].astype(str)

    print(f"\n=== Top {len(top_df)} by Train R²（Test R² > {test_thr}） ===")
    if top_df.empty:
        print("※ 沒有符合條件的模型，請下修 test_thr 或調整搜尋範圍。")
    else:
        with pd.option_context('display.max_colwidth', 1000):
            print(
                top_df[[
                    'Scaler', 'Features',
                    'n_estimators', 'max_depth', 'RF_seed',
                    'R2_train', 'R2_test', 'RMSE_test', 'MAE_test'
                ]].to_string(index=False,
                             formatters={
                                 'R2_train':  '{:.6f}'.format,
                                 'R2_test':   '{:.6f}'.format,
                                 'RMSE_test': '{:.4f}'.format,
                                 'MAE_test':  '{:.4f}'.format
                             })
            )
