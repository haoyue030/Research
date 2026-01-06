'''
單層測試 特徵組合
'''

############################################## 特徵組合選擇 ########################
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler,RobustScaler
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
    '風速', '氣壓',
    'wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos',
    '波高_YJ', '降雨_YJ', '暴風半徑_YJ',
    '潮位_BC2',
    '波能_log1p', '功率_log1p',
    '尖峰週期_BC'
]

# ----------------------------------------
# 2.5 ★ 先對「整份資料」做標準化
# ----------------------------------------
X_full_raw = df[candidate].values
scaler_x_full = RobustScaler().fit(X_full_raw)                 # ★ 全資料 fit
scaler_y_full = RobustScaler().fit(y_all.reshape(-1,1))        # ★ 全資料 fit
X_full_std = scaler_x_full.transform(X_full_raw)
y_full_std = scaler_y_full.transform(y_all.reshape(-1,1)).ravel()

# 建立特徵索引對應
feat_to_idx = {f:i for i, f in enumerate(candidate)}

# ----------------------------------------
# 3. 網路架構列表 (2 層，每層 1~20 個神經元)
# ----------------------------------------
architectures = [(6,), (12,), (20,)]

# ----------------------------------------
# 4. 特徵子集大小範圍
# ----------------------------------------
min_features = 3
max_features = 13

# ----------------------------------------
# 5. 遍歷特徵子集與網路架構（先全域標準化→再切分作單次評估）
# ----------------------------------------
results = []

for r in range(min_features, max_features + 1):
    for feats in combinations(candidate, r):
        # 保證 sin/cos 成對
        if ('wind_dir_sin' in feats) ^ ('wind_dir_cos' in feats): 
            continue
        if ('wave_dir_sin' in feats) ^ ('wave_dir_cos' in feats): 
            continue

        X_all_std = X_full_std[:, [feat_to_idx[f] for f in feats]]

        # 同步切分（標準化後的 X、y 以及原始 y）
        X_tr_s, X_te_s, y_tr_s, y_te_s, y_tr, y_te = train_test_split(
            X_all_std, y_full_std, y_all, test_size=0.3, random_state=42
        )

        for arch in architectures:
            mlp_params = dict(
                hidden_layer_sizes=arch,
                activation='relu',
                solver='adam',
                batch_size=16,
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=10,
                max_iter=200
            )
            mlp = MLPRegressor(**mlp_params, random_state=0)
            mlp.fit(X_tr_s, y_tr_s)

            # 單次切分（用全域 y-scaler 反標準化回原單位再評估）
            y_tr_pred = scaler_y_full.inverse_transform(mlp.predict(X_tr_s).reshape(-1,1)).ravel()
            y_te_pred = scaler_y_full.inverse_transform(mlp.predict(X_te_s).reshape(-1,1)).ravel()

            r2_train = r2_score(y_tr, y_tr_pred)
            r2_test  = r2_score(y_te, y_te_pred)
            rmse_test = np.sqrt(mean_squared_error(y_te, y_te_pred))

            results.append({
                'Features':     feats,
                'Num':          r,
                'Architecture': arch,
                'R2_train':     r2_train,
                'R2_test':      r2_test,
                'RMSE_test':    rmse_test
            })

# ----------------------------------------
# 6. 整理輸出：Top K by Train R²（Test R² > 門檻），美化列印 + 輸出 CSV
# ----------------------------------------
res_df = pd.DataFrame(results)

test_thr = 0.50   # Test R² 下限
top_k   = 10      # 取前幾名

# 只取需要欄位（保留 tuple 型態，稍後再美化）
base_df = res_df[['Features', 'Architecture', 'R2_train', 'R2_test', 'RMSE_test']].copy()

# 篩選 & 排序
top_df = (
    base_df.loc[base_df['R2_test'] > test_thr]
           .sort_values(['R2_train', 'R2_test', 'RMSE_test'],
                        ascending=[False,     False,      True])
           .head(top_k)
           .copy()
)

# ===== 美化顯示 =====
# Features: ('a','b','c') 的風格；Architecture: "(h1, h2)"
top_df['Features'] = top_df['Features'].apply(
    lambda t: '(' + ', '.join(f"'{f}'" for f in t) + ')'
)
top_df['Architecture'] = top_df['Architecture'].apply(
    lambda a: f"({a[0]})"
)

# 數字格式
fmt = {
    'R2_train':  '{:.6f}'.format,
    'R2_test':   '{:.6f}'.format,
    'RMSE_test': '{:.4f}'.format
}

print(f"\n=== Top {len(top_df)} by Train R²（Test R² > {test_thr}） ===")
if top_df.empty:
    print("※ 沒有符合條件的模型，請下修 test 門檻或調整搜尋範圍。")
else:
    # 提高欄寬，避免特徵被截斷
    with pd.option_context('display.max_colwidth', 1000):
        print(top_df.to_string(index=False, formatters=fmt))

# ============================================
# 7. 匯出結果（全部 + Top）
# ============================================
out_all = r'D:\OneDrive\桌面\新模型\MLP\MLP_singlelayer_all_results.csv'
out_top = r'D:\OneDrive\桌面\新模型\MLP\MLP_singlelayer_top_results.csv'

# ① 全部結果：先把 tuple 欄位轉成好讀字串再輸出
res_out = res_df.copy()
res_out['Features'] = res_out['Features'].apply(lambda t: '(' + ', '.join(t) + ')')
res_out['Architecture'] = res_out['Architecture'].apply(lambda a: '(' + ', '.join(map(str, a)) + ')')
res_out.to_csv(out_all, index=False, encoding='utf-8-sig')

# ② Top 結果：你已經美化過 top_df，直接輸出
top_df.to_csv(out_top, index=False, encoding='utf-8-sig')

print("\n--- 已匯出 ---")
print("全部結果：", out_all)
print("Top結果：", out_top)
