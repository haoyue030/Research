'''
RobustScaler、特徵組合篩選：StandardScaler結果跟RobustScaler一樣
'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer, RobustScaler,StandardScaler
from scipy.stats import boxcox
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import itertools

# ==========================
# 1. 讀取並前處理
# ==========================
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)
x = df['尖峰週期']
df['尖峰週期_BC'] = boxcox(x - x.min() + 1e-6, 2.4217)

y = df['y'].values

candidate = [
    '風速', '氣壓',
    'wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos',
    '波高_YJ','降雨_YJ','暴風半徑_YJ',
    '潮位_BC2',
    '波能_log1p','功率_log1p',
    '尖峰週期_BC'
]

# ==========================
# 2. 先標準化全資料 → 再固定切分
# ==========================
# ★ 重點：先對「整份候選特徵矩陣」做 RobustScaler
X_full_raw = df[candidate].values
scaler = RobustScaler().fit(X_full_raw)
X_full_std = scaler.transform(X_full_raw)

# 建立欄位名稱到索引的對應，方便後面快速抽取子集合
feat_to_idx = {f: i for i, f in enumerate(candidate)}

indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=0)

# ==========================
# 3. 排列組合並建模（已先全域標準化；模型不再做縮放）
# ==========================
results = []
min_features = 1
max_features = len(candidate)

for k in range(min_features, max_features + 1):
    for combo in itertools.combinations(candidate, k):
        # 保證方向特徵成對
        if ('wind_dir_sin' in combo) ^ ('wind_dir_cos' in combo):   continue
        if ('wave_dir_sin' in combo) ^ ('wave_dir_cos' in combo):   continue

        cols_idx = [feat_to_idx[f] for f in combo]
        X_all   = X_full_std[:, cols_idx]      # 從「已標準化矩陣」抽子欄位
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        r2_train = model.score(X_train, y_train)
        r2_test  = model.score(X_test,  y_test)

        results.append({
            'features': combo,
            'n_features': len(combo),
            'r2_train': r2_train,
            'r2_test':  r2_test
        })

# ==========================
# 4. 整理並印出 Top 10 by Train R²（Test R² > 0.4）
# ==========================
results_df = pd.DataFrame(results)
test_thr = 0.50
top_k = 10

if results_df.empty:
    print("⚠️ 沒有任何結果（results 為空）。")
else:
    # 過濾 Test R² 門檻，再依 Train R² 排序；同分用 Test R² 下降比；再同分可偏好較少特徵
    filt = (results_df
            .loc[results_df['r2_test'] > test_thr]
            .sort_values(['r2_train', 'r2_test', 'n_features'],
                         ascending=[False,     False,       True])
            .head(top_k)
            .copy())

    if filt.empty:
        print(f"⚠️ 沒有組合符合 Test R² > {test_thr:.2f} 的門檻。")
    else:
        # 美化 features 顯示
        filt['features'] = filt['features'].apply(lambda t: '(' + ', '.join(f"'{f}'" for f in t) + ')')
        # 數字格式
        fmt = {'r2_train': '{:.6f}'.format, 'r2_test': '{:.6f}'.format}
        pd.set_option('display.max_colwidth', 1000)
        print(f"\n✅ Top {len(filt)} by Train R²（Test R² > {test_thr:.2f}）")
        print(filt[['features','n_features','r2_train','r2_test']]
              .reset_index(drop=True)
              .to_string(index=False, formatters=fmt))
