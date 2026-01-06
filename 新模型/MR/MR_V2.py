# ============================================
# 線性回歸（全部特徵標準化）+ PCA 僅做重要性檢查（不進模型）
# ============================================
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy.stats import boxcox, shapiro

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 顯示中文
plt.rcParams['axes.unicode_minus'] = False

# ==========================
# 1. 讀取並前處理
# ==========================
df = pd.read_excel(r'D:\OneDrive\桌面\新模型\0805.xlsx', sheet_name='Sheet1')

# 1.2 特徵與目標
best_features = ['氣壓',
    'wave_dir_sin','wave_dir_cos',
    '波高','降雨',
    '尖峰週期'
]
X = df[best_features].copy()
y = df['y'].copy()

# （可選）若有缺失值，先丟棄
data = pd.concat([X, y], axis=1).dropna()
X = data[best_features]
y = data['y']

# ==========================
# 2. 切分 + 全部特徵標準化（僅用訓練集 fit）
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()
X_train_std = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=best_features, index=X_train.index
)
X_test_std = pd.DataFrame(
    scaler.transform(X_test),
    columns=best_features, index=X_test.index
)
# ==========================
# 3. VIF（以「訓練集標準化後」特徵計算）
# ==========================
Xc = sm.add_constant(X_train_std)
vif_tbl = pd.DataFrame({
    'Variable': ['const'] + list(X_train_std.columns),
    'VIF': [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])]
})
print("✅ 訓練集（標準化後）VIF：")
print(vif_tbl.round(4).to_string(index=False), "\n")

# ==========================
# 4. 建立並擬合模型（只用標準化後特徵；不含 PCA）
# ==========================
lr = LinearRegression()
lr.fit(X_train_std, y_train)

print(f"✅ 截距 β0 = {lr.intercept_:.6f}")
for name, coef in zip(best_features, lr.coef_):
    print(f"✅ 係數 β({name}) = {coef:.6f}")

# ==========================
# 5. 評估與視覺化
# ==========================
def evaluate_and_plot(model, Xmat, y_true, name):
    """對 (Xmat, y_true) 評估並繪製診斷圖。"""
    y_pred = model.predict(Xmat)
    res = y_true - y_pred

    # 指標
    mse   = mean_squared_error(y_true, y_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)

    print(f"\n=== {name} Set ===")
    print(f"MSE:  {mse:.4e}")
    print(f"RMSE: {rmse:,.4f}")
    print(f"MAE:  {mae:,.4f}")
    print(f"R²:   {r2:.6f}")

    # True vs Pred 散點 + 45°線
    plt.figure(figsize=(8,7))
    plt.scatter(y_true, y_pred, alpha=0.7, edgecolor='k', s=150, label='estimate vs True')
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--', linewidth=1.5, label='45° Reference')
    plt.xlabel('True y (m³)', fontsize=18)
    plt.ylabel('estimate y (m³)', fontsize=18)
    plt.title(f'{name} Set', fontsize=20)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.show()

    # Shapiro-Wilk 正態性檢定（殘差）
    res = np.asarray(res).ravel()
    if res.shape[0] > 5000:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(res.shape[0], 5000, replace=False)
        res_for_test = res[idx]
    else:
        res_for_test = res
    stat, p = shapiro(res_for_test)
    print(f"Shapiro-Wilk ({name} Residuals): Statistic={stat:.4f}, p-value={p:.4f}")
    print("➡️ 殘差近似正態分布\n" if p > 0.05 else "➡️ 殘差不符合正態分布\n")

def plot_scatter_true_vs_pred_topk(y_true, y_pred, title, top_k=5, fmt="{:,.0f}"):
    """真實 vs 預測，僅標註 |殘差| 最大的前 K 點"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, s=150, edgecolor='k', label='estimate vs True')

    lims = [np.min([y_true.min(), y_pred.min()]),
            np.max([y_true.max(), y_pred.max()])]
    plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')

    resid = np.abs(y_true - y_pred)
    idx = np.argsort(resid)[-top_k:]

    for i in idx:
        try:
            true_str = fmt.format(y_true[i])
        except Exception:
            true_str = f"{y_true[i]:.2f}"
        delta_val = resid[i]
        plt.annotate(
            f"{true_str}\nΔ={delta_val:,.0f}",
            xy=(y_true[i], y_pred[i]),
            xytext=(8, -8), textcoords="offset points",
            fontsize=14,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", lw=1, alpha=0.7)
        )
        plt.scatter([y_true[i]], [y_pred[i]], s=300, facecolors='none', edgecolors='red', linewidths=2)

    plt.xlabel('True y (m³)', fontsize=18)
    plt.ylabel('estimate y (m³)', fontsize=18)
    plt.title(title, fontsize=20)
    plt.xlim(lims); plt.ylim(lims)
    plt.xticks(fontsize=14); plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box')
    plt.legend(fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# === 評估與列表 ===
y_pred_train = lr.predict(X_train_std)
y_pred_test  = lr.predict(X_test_std)

evaluate_and_plot(lr, X_train_std, y_train, 'Train')
evaluate_and_plot(lr, X_test_std,  y_test,  'Test')

train_pred_tbl = pd.DataFrame({
    'True_y': y_train.reset_index(drop=True),
    'Pred_y': pd.Series(y_pred_train)
})
train_pred_tbl['Residual']  = train_pred_tbl['True_y'] - train_pred_tbl['Pred_y']
train_pred_tbl['Abs_Error'] = train_pred_tbl['Residual'].abs()
print("\n=== Train Set: True vs Pred ===")
print(train_pred_tbl.round(4).to_string(index=False))

test_pred_tbl = pd.DataFrame({
    'True_y': y_test.reset_index(drop=True),
    'Pred_y': pd.Series(y_pred_test)
})
test_pred_tbl['Residual']  = test_pred_tbl['True_y'] - test_pred_tbl['Pred_y']
test_pred_tbl['Abs_Error'] = test_pred_tbl['Residual'].abs()
print("\n=== Test Set: True vs Pred ===")
print(test_pred_tbl.round(4).to_string(index=False))

plot_scatter_true_vs_pred_topk(y_train, y_pred_train, 'Train Set')
plot_scatter_true_vs_pred_topk(y_test,  y_pred_test,  'Test Set')

# ==========================
# 6. PCA（診斷用；不進模型）
# ==========================
from sklearn.decomposition import PCA

# 對齊你的命名
X_train_s = X_train_std.copy()
feature_cols = best_features

pca = PCA(n_components=None, svd_solver='full', random_state=42).fit(X_train_s)

# --- PC1 的特徵值 / 解釋率 ---
pc1_eigenvalue = pca.explained_variance_[0]          # λ1
pc1_ratio      = pca.explained_variance_ratio_[0]    # PC1 explained variance ratio

print("\n=== PCA（基於 X_train_s）===")
print(f"特徵數: {X_train_s.shape[1]}")
print(f"PC1 特徵值 (Eigenvalue): {pc1_eigenvalue:.6f}")
print(f"PC1 解釋變異比例: {pc1_ratio*100:.2f}%")

# --- PC1 特徵向量（loading，依 |loading| 由大到小）---
pc1_vector = pca.components_[0]   # shape = (n_features,)
pc1_loadings = pd.Series(pc1_vector, index=feature_cols, name='PC1_loading') \
                 .sort_values(key=np.abs, ascending=False)
print("\nPC1 特徵向量（loading，依 |loading| 由大到小）：")
print(pc1_loadings.round(6).to_string())

# --- 全部主成分摘要 ---
evr = pca.explained_variance_ratio_
eig_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(evr))],
    'Eigenvalue': pca.explained_variance_,
    'Explained_%': evr * 100,
    'Cumulative_%': np.cumsum(evr) * 100
})
print("\n全部主成分摘要:")
print(eig_df.round(4).to_string(index=False))

# --- 全部特徵向量（loadings 矩陣）---
loadings_df = pd.DataFrame(
    pca.components_.T, index=feature_cols,
    columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])]
)
print("\n全部特徵向量:")
print(loadings_df.round(6).to_string())

# （可選）若想加總各 PC 的加權平方 loading 作為整體特徵重要性：
# importance_i = Σ_j ( components[j,i]^2 * evr[j] )
# pca_importance = pd.Series((pca.components_**2).T @ evr,
#                            index=feature_cols, name='PCA_importance_%') * 100
# print("\n🔎 PCA 特徵重要性（%）：")
# print(pca_importance.sort_values(ascending=False).round(2).to_string())
