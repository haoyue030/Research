import os
import numpy as np
import pandas as pd
from itertools import combinations, product
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------- 1) 讀取 & 轉換 ----------
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2

for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

x = df['尖峰週期']
df['尖峰週期_BC'] = boxcox(x - x.min() + 1e-6, 2.4217)

y_all = df['y'].values

feature_columns = [
'氣壓', 'wave_dir_sin', 'wave_dir_cos'
]

# ---------- 2) 縮放（先全域 fit→transform） ----------
X_full_raw = df[feature_columns].values
scalers = {'Robust': RobustScaler().fit(X_full_raw)}  # 可再加 'Standard': StandardScaler().fit(X_full_raw)
X_scaled = {name: sc.transform(X_full_raw) for name, sc in scalers.items()}
feat_to_idx = {f: i for i, f in enumerate(feature_columns)}

# 固定一次切分（全組合共用）
idx = np.arange(len(df))
train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=0)

# ---------- 3) 搜尋參數 ----------
n_estimators_range = list(range(10, 251, 10))
max_depth_range = [3,4,5,6,7,8,9]
RANDOM_STATE = 42

import matplotlib.pyplot as plt

def plot_r2_heatmaps(Xm, y, cols, show=True):
    """
    只顯示 R² 熱力圖（train / test / 差值），不做任何存檔。
    差值定義：R2_test - R2_train
    x 軸：max_depth；y 軸：n_estimators
    """
    Xs = Xm[:, cols]
    X_tr, X_te = Xs[train_idx], Xs[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    r2_train_mat = np.empty((len(n_estimators_range), len(max_depth_range)))
    r2_test_mat  = np.empty((len(n_estimators_range), len(max_depth_range)))

    for i, n_est in enumerate(n_estimators_range):
        for j, depth in enumerate(max_depth_range):
            rf = RandomForestRegressor(
                n_estimators=n_est,
                max_depth=depth,
                random_state=RANDOM_STATE
            )
            rf.fit(X_tr, y_tr)
            r2_train_mat[i, j] = r2_score(y_tr, rf.predict(X_tr))
            r2_test_mat[i, j]  = r2_score(y_te, rf.predict(X_te))

    # 轉成 DataFrame（列：n_estimators；欄：max_depth）
    df_r2_train = pd.DataFrame(r2_train_mat, index=n_estimators_range, columns=max_depth_range)
    df_r2_test  = pd.DataFrame(r2_test_mat,  index=n_estimators_range, columns=max_depth_range)
    df_r2_diff  = df_r2_test - df_r2_train   # 新增：差值熱力圖（R2_test - R2_train）

    # 用 matplotlib 畫三張熱力圖
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(df_r2_train.values, aspect='auto', origin='lower')
    axes[0].set_title("R² (train)")
    axes[0].set_xlabel("max_depth")
    axes[0].set_ylabel("n_estimators")
    axes[0].set_xticks(range(len(max_depth_range)))
    axes[0].set_xticklabels(max_depth_range)
    axes[0].set_yticks(range(len(n_estimators_range)))
    axes[0].set_yticklabels(n_estimators_range)
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(df_r2_test.values, aspect='auto', origin='lower')
    axes[1].set_title("R² (test)")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("n_estimators")
    axes[1].set_xticks(range(len(max_depth_range)))
    axes[1].set_xticklabels(max_depth_range)
    axes[1].set_yticks(range(len(n_estimators_range)))
    axes[1].set_yticklabels(n_estimators_range)
    fig.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(df_r2_diff.values, aspect='auto', origin='lower')
    axes[2].set_title("R² (test - train)")
    axes[2].set_xlabel("max_depth")
    axes[2].set_ylabel("n_estimators")
    axes[2].set_xticks(range(len(max_depth_range)))
    axes[2].set_xticklabels(max_depth_range)
    axes[2].set_yticks(range(len(n_estimators_range)))
    axes[2].set_yticklabels(n_estimators_range)
    fig.colorbar(im2, ax=axes[2])

    if show:
        plt.show()
    else:
        plt.close(fig)


cols = [feat_to_idx[f] for f in feature_columns]
plot_r2_heatmaps(X_scaled['Robust'], y_all, cols, show=True)
