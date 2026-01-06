import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

# 設定中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取資料
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx')

# 2. 特徵列表
features = [
    '風速', '氣壓', '波高', '降雨', '潮位', '暴風半徑','功率','波能','尖峰週期',
    'wind_dir_sin', 'wind_dir_cos', 'wave_dir_sin', 'wave_dir_cos'
]

# 3. 建立輸出資料夾
output_dir = r'D:\OneDrive\桌面\新模型\QQ_plots'
os.makedirs(output_dir, exist_ok=True)

# 4. 繪製並儲存 Q–Q Plot
for feat in features:
    data = df[feat].dropna()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    sm.qqplot(data, line='s', ax=ax)
    
    ax.set_title(f'{feat} QQ_plot', fontsize=16)
    plt.tight_layout()
    
    # 儲存圖檔
    filename = f"{feat}_QQ_plot.png"
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath)
    plt.close(fig)

print(f"所有 Q–Q 圖已儲存在：{output_dir}")


# ====== 5.（新增）做特徵轉換 ======
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats import boxcox

# 建立轉換器（Yeo–Johnson）
pt = PowerTransformer(method='yeo-johnson', standardize=False)

# (a) Yeo–Johnson：波高、降雨、暴風半徑
for col in ['波高', '降雨', '暴風半徑']:
    if col in df.columns:
        df[col + '_YJ'] = pt.fit_transform(df[[col]])

# (b) Box–Cox λ=2（平方）：潮位（需先位移到正值）
if '潮位' in df.columns:
    df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2

# (c) log1p：波能、功率（先位移避免非正值）
for col in ['波能', '功率']:
    if col in df.columns:
        df[col + '_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

# (d) Box–Cox λ≈2.42：尖峰週期（需先位移到正值）
if '尖峰週期' in df.columns:
    x_pos = df['尖峰週期'] - df['尖峰週期'].min() + 1e-6
    df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

# ====== 6.（新增）繪製【轉換後】Q–Q 圖並儲存 ======
transformed_cols = []
transformed_cols += [c for c in ['波高_YJ', '降雨_YJ', '暴風半徑_YJ'] if c in df.columns]
transformed_cols += [c for c in ['潮位_BC2', '波能_log1p', '功率_log1p', '尖峰週期_BC'] if c in df.columns]

output_dir_trans = r'D:\OneDrive\桌面\新模型\QQ_plots_transformed'
os.makedirs(output_dir_trans, exist_ok=True)

for col in transformed_cols:
    data = df[col].dropna()
    if data.empty:
        continue

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    sm.qqplot(data, line='s', ax=ax)
    ax.set_title(f'{col} QQ_plot（轉換後）', fontsize=16)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir_trans, f"{col}_QQ_plot.png"))
    plt.close(fig)

print(f"所有【轉換後】Q–Q 圖已儲存在：{output_dir_trans}")
