import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 1) 讀取數據
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

# 2) 數據轉換
for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)
x = df['尖峰週期']
x_pos = x - x.min() + 1e-6
df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

# 3) 特徵與目標
feat_names = ['氣壓', 'wave_dir_sin', 'wave_dir_cos'
              ]
X_raw = df[feat_names].values
y = df['y'].values

# 4) ★ 切分前全域標準化（RobustScaler）
scaler = RobustScaler().fit(X_raw)
X_std = scaler.transform(X_raw)

# 5) 切分（用標準化後的特徵）
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.3, random_state=42
)

# 6) 訓練 RF
model_rf = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42)
model_rf.fit(X_train, y_train)

# --- Test Set 評估 ---
y_pred_test = model_rf.predict(X_test)
res_test    = y_test - y_pred_test
mse_t  = mean_squared_error(y_test, y_pred_test)
rmse_t = np.sqrt(mse_t)
mae_t  = mean_absolute_error(y_test, y_pred_test)
r2_t   = r2_score(y_test, y_pred_test)
print("=== Test Set ===")
print(f"MSE:  {mse_t:.4f}, RMSE: {rmse_t:.4f}, MAE: {mae_t:.4f}, R²: {r2_t:.4f}")

# --- Train Set 評估 ---
y_pred_train = model_rf.predict(X_train)
res_train     = y_train - y_pred_train
mse_tr  = mean_squared_error(y_train, y_pred_train)
rmse_tr = np.sqrt(mse_tr)
mae_tr  = mean_absolute_error(y_train, y_pred_train)
r2_tr   = r2_score(y_train, y_pred_train)
print("\n=== Train Set ===")
print(f"MSE:  {mse_tr:.4f}, RMSE: {rmse_tr:.4f}, MAE: {mae_tr:.4f}, R²: {r2_tr:.4f}")

# 7) 10-fold CV
scores = cross_val_score(model_rf, X_std, y, cv=10, scoring='r2')
print("\n10-fold CV R² 平均:", scores.mean())
print("各 fold R²:", scores)

# 8) 視覺化：殘差分布
plt.figure(figsize=(8,4))
sns.histplot(res_train, bins=20, kde=True, color='tab:blue', edgecolor='black', alpha=0.6)
plt.title('Train Residual Distribution'); plt.xlabel('Residual'); plt.ylabel('Frequency')
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
sns.histplot(res_test, bins=20, kde=True, color='tab:red', edgecolor='black', alpha=0.6)
plt.title('Test Residual Distribution'); plt.xlabel('Residual'); plt.ylabel('Frequency')
plt.tight_layout(); plt.show()

# 9) True vs Pred（Test）
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_test, alpha=0.7, s=150, edgecolor='k', label='Predicted vs True')
lims = [np.min([y_test.min(), y_pred_test.min()]), np.max([y_test.max(), y_pred_test.max()])]
plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')
plt.xlabel('True y (m³)', fontsize=18); plt.ylabel('Predicted y (m³)', fontsize=18)
plt.title('Test Set', fontsize=20)
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
plt.xlim(lims); plt.ylim(lims)
plt.gca().set_aspect('equal', 'box'); plt.legend(fontsize=18); plt.grid(True)
plt.tight_layout(); plt.show()

# 10) True vs Pred（Train）
plt.figure(figsize=(7, 7))
plt.scatter(y_train, y_pred_train, alpha=0.7, s=150, edgecolor='k', label='Predicted vs True')
lims = [np.min([y_train.min(), y_pred_train.min()]), np.max([y_train.max(), y_pred_train.max()])]
plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')
plt.xlabel('True y (m³)', fontsize=18); plt.ylabel('Predicted y (m³)', fontsize=18)
plt.title('Train Set', fontsize=20)
plt.xticks(fontsize=16); plt.yticks(fontsize=16)
plt.xlim(lims); plt.ylim(lims)
plt.gca().set_aspect('equal', 'box'); plt.legend(fontsize=18); plt.grid(True)
plt.tight_layout(); plt.show()

# 11) 列出訓練集 True vs Pred（含殘差）
train_pred_tbl = pd.DataFrame({'True_y': y_train, 'Pred_y': y_pred_train})
train_pred_tbl['Residual']  = train_pred_tbl['True_y'] - train_pred_tbl['Pred_y']  # 與上方一致：True - Pred
train_pred_tbl['Abs_Error'] = train_pred_tbl['Residual'].abs()
print("\n=== Train Set: True vs Pred ===")
print(train_pred_tbl.round(4).to_string(index=False))

# 12) 真實 vs 預測（標註 |殘差| 最大 Top-K）
def plot_scatter_true_vs_pred_topk(y_true, y_pred, title, top_k=5, fmt="{:,.0f}"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, s=150, edgecolor='k', label='estimate vs True')
    lims = [np.min([y_true.min(), y_pred.min()]), np.max([y_true.max(), y_pred.max()])]
    plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')
    resid = np.abs(y_pred - y_true)
    idx = np.argsort(resid)[-top_k:]
    for i in idx:
        plt.annotate(
            f"{fmt.format(y_true[i])}\nΔ={resid[i]:,.0f}",
            xy=(y_true[i], y_pred[i]),
            xytext=(8, -8), textcoords="offset points",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", lw=1, alpha=0.7)
        )
        plt.scatter([y_true[i]],[y_pred[i]], s=300, facecolors='none', edgecolors='red', linewidths=2)
    plt.xlabel('True y (m³)', fontsize=18); plt.ylabel('estimate y (m³)', fontsize=18)
    plt.title(title,fontsize=20); plt.xlim(lims); plt.ylim(lims)
    plt.xticks(fontsize=16); plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box'); plt.legend(fontsize=18); plt.grid(True); plt.tight_layout(); plt.show()

plot_scatter_true_vs_pred_topk(y_train, y_pred_train, 'Train Set')
plot_scatter_true_vs_pred_topk(y_test,  y_pred_test,  'Test Set')

# 13) 特徵重要性
importances = model_rf.feature_importances_
imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=False)
print("\n=== Feature Importances ===")
print(imp_df.to_string(index=False))

plt.figure(figsize=(8,4))
sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')
plt.title('Feature Importances', fontsize=16)
plt.xlabel('Importance', fontsize=14); plt.ylabel('Feature', fontsize=14)
plt.yticks(fontsize=14); plt.xticks(fontsize=14)
plt.tight_layout(); plt.show()
