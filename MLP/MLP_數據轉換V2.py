# ======================================
# MLP — 無學習曲線版本（只保留最終模型 + 評估散點圖）
# ======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------- Matplotlib 中文 ----------
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 路徑/基本設定 ----------
FILE = r'D:\OneDrive\桌面\新模型\0805.xlsx'
SHEET = 'Sheet1'
FEATURE_COLS = [
    '風速','氣壓','wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos','暴風半徑_YJ','尖峰週期_BC'
]
HIDDEN = (20, 20)
SEED_SPLIT = 0
SEED_MLP   = 42

# ---------- 函式：評估 + 還原尺度 ----------
def eval_and_inverse(model, Xs, y_true, y_scaler, name='Set'):
    y_pred_s = model.predict(Xs)
    y_pred   = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()

    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)

    print(f"--- {name} ---")
    print(f"MSE : {mse:.4e}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"R²  : {r2:.6f}\n")

    return y_pred

# ---------- 函式：散點圖（標註殘差 TopK） ----------
def scatter_topk(y_true, y_pred, title, top_k=5, fmt="{:,.0f}"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, s=150,
                edgecolor='k', label='estimate vs True')

    lims = [min(y_true.min(), y_pred.min()),
            max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')

    # resid = np.abs(y_pred - y_true)
    # idx = np.argsort(resid)[-top_k:]   # 殘差最大的 top_k 筆

    # for i in idx:
    #     plt.annotate(
    #         f"{fmt.format(y_true[i])}\nΔ={resid[i]:,.0f}",
    #         xy=(y_true[i], y_pred[i]),
    #         xytext=(8,-8), textcoords="offset points",
    #         fontsize=14,
    #         bbox=dict(boxstyle="round,pad=0.15",
    #                   fc="white", ec="red", lw=1, alpha=0.7)
    #     )
    #     plt.scatter([y_true[i]],[y_pred[i]],
    #                 s=300, facecolors='none',
    #                 edgecolors='red', linewidths=2)

    plt.xlabel('True y (m³)', fontsize=18)
    plt.ylabel('estimate y (m³)', fontsize=18)
    plt.title(title, fontsize=20)
    plt.xlim(lims); plt.ylim(lims)
    plt.xticks(fontsize=16); plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box')
    plt.legend(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ---------- 讀檔 + 特徵工程 ----------
df = pd.read_excel(FILE, sheet_name=SHEET)

# 依你原本設定做轉換
for col in ['波高','降雨','暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6)**2

for col in ['波能','功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

df['尖峰週期_BC'] = boxcox(
    df['尖峰週期'] - df['尖峰週期'].min() + 1e-6,
    2.4217
)

X_raw = df[FEATURE_COLS].values
y_raw = df['y'].values

# ---------- 全域標準化（注意：有資料外洩風險） ----------
scaler_x = RobustScaler().fit(X_raw)
scaler_y = RobustScaler().fit(y_raw.reshape(-1,1))

X_s = scaler_x.transform(X_raw)
y_s = scaler_y.transform(y_raw.reshape(-1,1)).ravel()

# ---------- 固定切分 ----------
X_train_s, X_test_s, y_train, y_test, y_train_s, y_test_s = train_test_split(
    X_s, y_raw, y_s,
    test_size=0.3,
    random_state=SEED_SPLIT
)

# ---------- 訓練「最終模型」（只有這個，沒有學習曲線追蹤） ----------
final_mlp = MLPRegressor(
    hidden_layer_sizes=HIDDEN,
    activation='relu',
    solver='adam',
    batch_size=16,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=10,
    max_iter=200,
    random_state=SEED_MLP
)

final_mlp.fit(X_train_s, y_train_s)

# ---------- 評估 + 畫散點圖 ----------
y_pred_train = eval_and_inverse(final_mlp, X_train_s, y_train, scaler_y, "Train Set")
y_pred_test  = eval_and_inverse(final_mlp, X_test_s,  y_test,  scaler_y, "Test Set")

scatter_topk(y_train, y_pred_train, 'Train Set')
scatter_topk(y_test,  y_pred_test,  'Test Set')

