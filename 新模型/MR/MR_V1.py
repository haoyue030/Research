import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import itertools

# ==========================
# 1. 讀取並前處理
# ==========================
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')
y = df['y'].values

candidate = [
    '風速', '氣壓',
    'wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos',
    '波高','降雨','暴風半徑',
    '潮位',
    '波能','功率',
    '尖峰週期'
]

# R² 門檻（和註解一致）
R2_THRESH = 0.4

# ==========================
# 2. 固定切分訓練／測試集
# ==========================
indices = np.arange(len(df))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)

# ==========================
# 3. 排列組合並建模（含標準化）
# ==========================
results = []
min_features = 1
max_features = len(candidate)

for k in range(min_features, max_features + 1):
    for combo in itertools.combinations(candidate, k):
        # 保證方向特徵成對
        if ('wind_dir_sin' in combo) ^ ('wind_dir_cos' in combo):   continue
        if ('wave_dir_sin' in combo) ^ ('wave_dir_cos' in combo):   continue

        X_all   = df[list(combo)].values
        X_train = X_all[train_idx]
        X_test  = X_all[test_idx]
        y_train = y[train_idx]
        y_test  = y[test_idx]

        # 避免某些特徵在訓練集為常數造成 StandardScaler 除以 0
        if np.any(np.nanstd(X_train, axis=0) == 0):
            continue

        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ('scaler', StandardScaler(with_mean=True, with_std=True)),
            ('linreg', LinearRegression())
        ])

        pipe.fit(X_train, y_train)

        r2_train = r2_score(y_train, pipe.predict(X_train))
        r2_test  = r2_score(y_test,  pipe.predict(X_test))

        # 篩選條件：訓練／測試 R² 都要 > R2_THRESH
        if r2_train > R2_THRESH and r2_test > R2_THRESH:
            results.append({
                'features': combo,
                'r2_train': r2_train,
                'r2_test':  r2_test
            })

# ==========================
# 4. 整理並印出前 10 名
# ==========================
results_df = pd.DataFrame(results)

if not results_df.empty:
    results_top10 = results_df.sort_values('r2_test', ascending=False).head(10)
    pd.set_option('display.max_colwidth', None)
    print(f"✅ 訓練／測試 R² 均 > {R2_THRESH}，前 10 名：")
    print(results_top10.reset_index(drop=True))
else:
    print(f"⚠️ 沒有任何特徵組合同時符合訓練集與測試集 R² > {R2_THRESH} 的條件。")
