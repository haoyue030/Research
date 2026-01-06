############################################## 網路架構選擇 ########################
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler,StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ----------------------------------------
# 1. 讀取並轉換數據
# ----------------------------------------
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

# Yeo–Johnson 轉換
for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

# 潮位平方轉換
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2

# 波能、功率 log1p
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

# 尖峰週期 Box–Cox λ=2.4217
x = df['尖峰週期']; x_pos = x - x.min() + 1e-6
df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

y_all = df['y'].values

# ----------------------------------------
# 2. 候選特徵清單
# ----------------------------------------
candidate = [
'風速', '氣壓', 'wind_dir_sin', 'wind_dir_cos', 'wave_dir_sin', 'wave_dir_cos', '暴風半徑_YJ', '尖峰週期_BC'
]

# ----------------------------------------
# 2.5 ★ 先對「整份資料」做標準化
# ----------------------------------------
X_full_raw = df[candidate].values
scaler_x_full = StandardScaler().fit(X_full_raw)                 # ★ 全資料 fit
scaler_y_full = StandardScaler().fit(y_all.reshape(-1,1))        # ★ 全資料 fit
X_full_std = scaler_x_full.transform(X_full_raw)
y_full_std = scaler_y_full.transform(y_all.reshape(-1,1)).ravel()

# 建立特徵索引對應
feat_to_idx = {f:i for i, f in enumerate(candidate)}

# ----------------------------------------
# 3. 網路架構列表：加入一層的測試（10~25 個神經元）
arch_2layer = [(i, j) for i in range(12, 21) for j in range(12, 21)]  # e.g. (10,10) ~ (25,25)

architectures = arch_2layer

# ----------------------------------------
# 4. 固定特徵組合（直接用 candidate 全部）— 先鎖定與抽欄
# ----------------------------------------
fixed_feats = tuple(candidate)
NUM_FEATS = len(fixed_feats)

# 依固定特徵從已標準化矩陣抽欄
X_all_std = X_full_std[:, [feat_to_idx[f] for f in fixed_feats]]

# ----------------------------------------
# 5. 只遍歷網路架構（固定特徵）
# ----------------------------------------
results = []

# 固定一次切分，所有架構共用同一個 Train/Test
X_tr_s, X_te_s, y_tr_s, y_te_s, y_tr, y_te = train_test_split(
    X_all_std, y_full_std, y_all, test_size=0.3, random_state=0
)

for arch in architectures:
    mlp = MLPRegressor(
        hidden_layer_sizes=arch,
        activation='relu',
        solver='adam',
        batch_size=16,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        max_iter=200,
        random_state=42
    )
    mlp.fit(X_tr_s, y_tr_s)

    # 用全域 y-scaler 反標準化回原單位再評估
    y_tr_pred = scaler_y_full.inverse_transform(mlp.predict(X_tr_s).reshape(-1, 1)).ravel()
    y_te_pred = scaler_y_full.inverse_transform(mlp.predict(X_te_s).reshape(-1, 1)).ravel()

    r2_train = r2_score(y_tr, y_tr_pred)
    r2_test  = r2_score(y_te, y_te_pred)
    rmse_test = float(np.sqrt(mean_squared_error(y_te, y_te_pred)))

    results.append({
        'Features':     fixed_feats,
        'Num':          len(fixed_feats),
        'Architecture': arch,
        'R2_train':     r2_train,
        'R2_test':      r2_test,
        'RMSE_test':    rmse_test
    })

# ----------------------------------------
# 6. 篩選 & 顯示：Top 10 by Train R²（且 Test R² > 門檻）
# ----------------------------------------
res_df = pd.DataFrame(results)

test_thr = 0.4   # Test R² 下限
top_k = 10        # 取前幾名

# 先過濾 Test R² 門檻，再用 Train R² 排序（同分再看 Test R² 高、RMSE 低）
top_df = (
    res_df.loc[res_df['R2_test'] > test_thr].copy()
            .sort_values(['R2_train', 'R2_test', 'RMSE_test'],
                         ascending=[False,   False,     True])
            .head(top_k)
)

# 美化 Features 顯示
top_df['Features'] = top_df['Features'].apply(
    lambda t: '(' + ', '.join(f"'{f}'" for f in t) + ')'
)

print(f"\n=== Top {len(top_df)} by Train R²（Test R² > {test_thr}） ===")
if top_df.empty:
    print("※ 沒有符合條件的模型，請下修 test 門檻或調整搜尋範圍。")
else:
    print(top_df[['Features','Architecture','R2_train','R2_test','RMSE_test']]
          .to_string(index=False))
