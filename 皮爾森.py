import pandas as pd
import numpy as np
from scipy.stats import pearsonr
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
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ==========================
# 2. 用你的資料產出「相關係數表 + 共變數表」（下三角、含星號）
# ==========================
target_col = "y"
cols = candidate + [target_col]

data = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

# --- Pearson r 與 p-value ---
corr = data.corr(method="pearson")

pval = pd.DataFrame(np.ones((len(cols), len(cols))), index=cols, columns=cols)
for i in range(len(cols)):
    for j in range(i + 1, len(cols)):
        r, p = pearsonr(data.iloc[:, i], data.iloc[:, j])
        pval.iat[i, j] = p
        pval.iat[j, i] = p

def sig_star(p):
    # 你截圖是 * / **，我用常見門檻；若只要到 **，保留前兩行即可
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""

# --- 下三角相關係數（上三角留白、對角=1）---
corr_tri = pd.DataFrame("", index=cols, columns=cols, dtype=object)
for i in range(len(cols)):
    for j in range(len(cols)):
        if j > i:
            corr_tri.iat[i, j] = ""
        elif i == j:
            corr_tri.iat[i, j] = "1"
        else:
            r = corr.iat[i, j]
            corr_tri.iat[i, j] = f"{r:.6f}{sig_star(pval.iat[i, j])}"

# --- 共變數（下三角；上三角留白）---
cov = data.cov()

cov_tri = pd.DataFrame("", index=cols, columns=cols, dtype=object)
for i in range(len(cols)):
    for j in range(len(cols)):
        if j > i:
            cov_tri.iat[i, j] = ""
        else:
            cov_tri.iat[i, j] = f"{cov.iat[i, j]:.6f}"

# ==========================
# 3. 輸出 Excel（兩張表）
# ==========================
out_path = r"D:\OneDrive\桌面\新模型\PearsonCorr_Cov_lowerTriangle.xlsx"
with pd.ExcelWriter(out_path) as writer:
    corr_tri.to_excel(writer, sheet_name="Pearson(lower+star)")
    cov_tri.to_excel(writer, sheet_name="Covariance(lower)")
    # 可選：也輸出完整矩陣與 p-value，方便你檢查
    corr.to_excel(writer, sheet_name="Pearson(full)")
    pval.to_excel(writer, sheet_name="p_value(full)")

print("已輸出：", out_path)
print("樣本數 n =", len(data))

