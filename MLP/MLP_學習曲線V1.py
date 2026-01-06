# ======================================
# MLP + 學習曲線（RAW 版：峰值/斜率/繪圖）— 無 Method A
# ======================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, RobustScaler,StandardScaler
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
N_EPOCHS   = 200

# ---------- 繪圖：只畫 RAW ----------
def plot_curve(ep, tr_vals, te_vals, ylabel, title,
               vline_epoch=None, vline_y=None, vline_label=None):
    plt.figure(figsize=(10, 5))
    plt.plot(ep, tr_vals, linewidth=2.2, label='Training')
    plt.plot(ep, te_vals, linewidth=2.2, label='Test')
    if vline_epoch is not None:
        plt.axvline(vline_epoch, ls='--', alpha=0.7, label=vline_label or f'@{vline_epoch}')
        if vline_y is not None:
            plt.scatter([vline_epoch], [vline_y], s=70, edgecolor='k', zorder=5)
    plt.xlabel('Epochs', fontsize=16);  plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=18)
    plt.xticks(fontsize=16); plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6); plt.legend(fontsize=16)
    plt.tight_layout(); plt.show()

def scatter_topk(y_true, y_pred, title, top_k=5, fmt="{:,.0f}"):
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.7, s=150, edgecolor='k', label='estimate vs True')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, 'r--', lw=2, label='45° Reference')
    resid = np.abs(y_pred - y_true)
    idx = np.argsort(resid)[-top_k:]
    for i in idx:
        plt.annotate(
            f"{fmt.format(y_true[i])}\nΔ={resid[i]:,.0f}",
            xy=(y_true[i], y_pred[i]), xytext=(8,-8), textcoords="offset points",
            fontsize=14, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="red", lw=1, alpha=0.7)
        )
        plt.scatter([y_true[i]],[y_pred[i]], s=300, facecolors='none', edgecolors='red', linewidths=2)
    plt.xlabel('True y (m³)', fontsize=18);  plt.ylabel('estimate y (m³)', fontsize=18)
    plt.title(title, fontsize=20); plt.xlim(lims); plt.ylim(lims)
    plt.xticks(fontsize=16); plt.yticks(fontsize=14)
    plt.gca().set_aspect('equal', 'box'); plt.legend(fontsize=18)
    plt.grid(True); plt.tight_layout(); plt.show()

def eval_and_inverse(model, Xs, y_true, y_scaler, name='Set'):
    y_pred_s = model.predict(Xs)
    y_pred   = y_scaler.inverse_transform(y_pred_s.reshape(-1,1)).ravel()
    mse  = mean_squared_error(y_true, y_pred);  rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred); r2   = r2_score(y_true, y_pred)
    print(f"--- {name} ---\nMSE : {mse:.4e}\nRMSE: {rmse:.4f}\nMAE : {mae:.4f}\nR²  : {r2:.6f}\n")
    return y_pred

def run_epoch_tracking(X_tr, y_tr_s, X_te, y_tr_orig, y_te_orig, y_scaler,
                       n_epochs=25, hidden=(20,20), seed=42, shuffle=False):
    mlp = MLPRegressor(hidden_layer_sizes=hidden, activation='relu',
                       solver='adam', batch_size=16,
                       early_stopping=False, max_iter=1, warm_start=True,
                       shuffle=shuffle, random_state=seed)
    def rmse(y_true, y_pred): return float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2_tr, r2_te, rmse_tr, rmse_te = [], [], [], []
    for _ in range(n_epochs):
        mlp.fit(X_tr, y_tr_s)
        yhat_tr = y_scaler.inverse_transform(mlp.predict(X_tr).reshape(-1,1)).ravel()
        yhat_te = y_scaler.inverse_transform(mlp.predict(X_te).reshape(-1,1)).ravel()
        r2_tr.append(r2_score(y_tr_orig, yhat_tr)); r2_te.append(r2_score(y_te_orig, yhat_te))
        rmse_tr.append(rmse(y_tr_orig, yhat_tr));   rmse_te.append(rmse(y_te_orig, yhat_te))
    ep = np.arange(1, n_epochs+1)
    return ep, np.array(r2_tr), np.array(r2_te), np.array(rmse_tr), np.array(rmse_te)

# ---------- 斜率方法（RAW；保留 Method B/C） ----------
def line_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return (np.polyfit(x, y, 1)[0]) if len(x) >= 2 else np.nan

def overall_slopes_after_test_extreme(ep, y_tr, y_te, mode='max', metric='', s_tag=''):
    ep, tr, te = map(lambda a: np.asarray(a, dtype=float), (ep, y_tr, y_te))
    idx = int(np.nanargmax(te)) if mode=='max' else int(np.nanargmin(te))
    if idx >= len(ep)-1:
        print(f"[{metric} {s_tag}] Test 極值在最後一個 epoch={int(ep[idx])}，沒有後續點可算整體斜率。")
        return None
    seg_ep, seg_tr, seg_te = ep[idx:], tr[idx:], te[idx:]
    return {
        'Metric': f'{metric} {s_tag}', 'Mode': 'max' if mode=='max' else 'min',
        'anchor_epoch': int(ep[idx]), 'n_points_in_segment': int(len(seg_ep)),
        'OLS_slope_Test_per_epoch':  float(line_slope(seg_ep, seg_te)),
        'OLS_slope_Train_per_epoch': float(line_slope(seg_ep, seg_tr)),
        'EndToEnd_slope_Test_per_epoch':  float((seg_te[-1]-seg_te[0])/(seg_ep[-1]-seg_ep[0])),
        'EndToEnd_slope_Train_per_epoch': float((seg_tr[-1]-seg_tr[0])/(seg_ep[-1]-seg_ep[0])),
    }

def stepwise_slopes_after_test_extreme(ep, y_tr, y_te, mode='max', metric='', s_tag=''):
    ep, tr, te = map(lambda a: np.asarray(a, dtype=float), (ep, y_tr, y_te))
    idx = int(np.nanargmax(te)) if mode=='max' else int(np.nanargmin(te))
    if idx >= len(ep)-1:
        print(f"[{metric} {s_tag}] Test 極值在最後一個 epoch={int(ep[idx])}，沒有後續段可算斜率。")
        return pd.DataFrame()
    rows, anchor = [], int(ep[idx])
    for j in range(idx, len(ep)-1):
        e_from, e_to = int(ep[j]), int(ep[j+1]); dE = e_to - e_from
        rows.append({
            'Metric': f'{metric} {s_tag}', 'Mode': 'max' if mode=='max' else 'min',
            'anchor_epoch': anchor, 'epoch_from': e_from, 'epoch_to': e_to, 'Δepoch': dE,
            'Test_value_from': float(te[j]),  'Test_value_to': float(te[j+1]),
            'Train_value_from': float(tr[j]), 'Train_value_to': float(tr[j+1]),
            'Test_Δvalue': float(te[j+1]-te[j]),  'Train_Δvalue': float(tr[j+1]-tr[j]),
            'Test_slope': float((te[j+1]-te[j])/dE), 'Train_slope': float((tr[j+1]-tr[j])/dE),
        })
    return pd.DataFrame(rows)

# ---------- 讀檔 + 特徵工程 ----------
df = pd.read_excel(FILE, sheet_name=SHEET)

for col in ['波高','降雨','暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6)**2
for col in ['波能','功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)
df['尖峰週期_BC'] = boxcox(df['尖峰週期'] - df['尖峰週期'].min() + 1e-6, 2.4217)

X_raw = df[FEATURE_COLS].values
y_raw = df['y'].values

# ---------- 全域標準化（依你需求；會有資料外洩風險） ----------
scaler_x = RobustScaler().fit(X_raw)
scaler_y = RobustScaler().fit(y_raw.reshape(-1,1))
X_s = scaler_x.transform(X_raw)
y_s = scaler_y.transform(y_raw.reshape(-1,1)).ravel()

# ---------- 固定切分 ----------
X_train_s, X_test_s, y_train, y_test, y_train_s, y_test_s = train_test_split(
    X_s, y_raw, y_s, test_size=0.3, random_state=SEED_SPLIT
)

# ---------- 訓練「最終模型」（非追蹤用） ----------
final_mlp = MLPRegressor(hidden_layer_sizes=HIDDEN, activation='relu',
                         solver='adam', batch_size=16,
                         early_stopping=True, validation_fraction=0.2,
                         n_iter_no_change=10, max_iter=200, random_state=SEED_MLP)
final_mlp.fit(X_train_s, y_train_s)

# 評估 + 散點（殘差 TopK）
y_pred_train = eval_and_inverse(final_mlp, X_train_s, y_train, scaler_y, "Train Set")
y_pred_test  = eval_and_inverse(final_mlp, X_test_s,  y_test,  scaler_y, "Test Set")
scatter_topk(y_train, y_pred_train, 'Train Set')
scatter_topk(y_test,  y_pred_test,  'Test Set')

# ---------- 追蹤學習曲線（逐 epoch；RAW） ----------
epochs, r2_tr, r2_te, rmse_tr, rmse_te = run_epoch_tracking(
    X_train_s, y_train_s, X_test_s, y_train, y_test, scaler_y,
    n_epochs=N_EPOCHS, hidden=HIDDEN, seed=SEED_MLP, shuffle=False
)

# 峰/谷（RAW）
idx_max_raw, ep_max_raw, val_max_raw = int(np.nanargmax(r2_te)), int(epochs[np.nanargmax(r2_te)]), float(np.nanmax(r2_te))
print(f"[RAW]  Test R² 最高點：{val_max_raw:.6f}，出現在 epoch = {ep_max_raw}")
idx_min_raw, ep_min_raw, val_min_raw = int(np.nanargmin(rmse_te)), int(epochs[np.nanargmin(rmse_te)]), float(np.nanmin(rmse_te))
print(f"[RAW]  Test RMSE 最低點：{val_min_raw:.4f}，出現在 epoch = {ep_min_raw}")

# 畫圖（RAW + 標註峰/谷）
plot_curve(epochs, r2_tr, r2_te, ylabel='R²',
           title='R² vs. Epochs (Train vs Test)',
           vline_epoch=ep_max_raw, vline_y=val_max_raw,
           vline_label=f'Test R² 最高點 {ep_max_raw} (raw)')

plot_curve(epochs, rmse_tr, rmse_te, ylabel='RMSE(m³)',
           title='RMSE vs. Epochs (Train vs Test)',
           vline_epoch=ep_min_raw, vline_y=val_min_raw,
           vline_label=f'Test RMSE 最低點 {ep_min_raw} (raw)')

# ---------- 斜率：只保留 Method B / Method C ----------
# 方法B：錨點 → 最後整段 OLS / 端到端
r2_overall   = overall_slopes_after_test_extreme(epochs, r2_tr,  r2_te,  mode='max', metric='R²',  s_tag='(RAW)')
rmse_overall = overall_slopes_after_test_extreme(epochs, rmse_tr, rmse_te, mode='min', metric='RMSE', s_tag='(RAW)')
summary_overall = pd.DataFrame([r2_overall, rmse_overall])

# 方法C：錨點之後相鄰段的接續斜率
r2_step_df   = stepwise_slopes_after_test_extreme(epochs, r2_tr,  r2_te,  mode='max', metric='R²',  s_tag='(RAW)')
rmse_step_df = stepwise_slopes_after_test_extreme(epochs, rmse_tr, rmse_te, mode='min', metric='RMSE', s_tag='(RAW)')

# 預覽（移除 Method A 的輸出）
pd.set_option('display.max_colwidth', 1000)
print("\n=== 方法B：整段斜率（錨點→最後 / RAW） ===")
print(summary_overall.to_string(index=False))
print("\n=== 方法C：接續斜率（R² / RAW）前 8 筆 ===")
print(r2_step_df.head(8).to_string(index=False)); print("..."); print(r2_step_df.tail(5).to_string(index=False))
print("\n=== 方法C：接續斜率（RMSE / RAW）前 8 筆 ===")
print(rmse_step_df.head(8).to_string(index=False)); print("..."); print(rmse_step_df.tail(5).to_string(index=False))

# ================================
# 兩種斜率法：統一繪圖（Method B/C，RAW）
# ================================
def plot_method_b_segment(
    ep, tr_raw, te_raw, summary_row, metric_name, ylabel,
    xticks_from_anchor=True, max_xticks=10, crop_left=True
):
    if summary_row is None:
        print(f"[{metric_name}] Method B 無 summary 可畫。")
        return
    anchor = int(summary_row['anchor_epoch'])
    try:
        start_idx = int(np.where(ep == anchor)[0][0])
    except IndexError:
        print(f"[{metric_name}] 找不到 anchor epoch={anchor}，略過 Method B。")
        return

    seg_ep = ep[start_idx:]
    seg_tr = tr_raw[start_idx:]
    seg_te = te_raw[start_idx:]

    # OLS 線（RAW）
    m_te, b_te = np.polyfit(seg_ep, seg_te, 1)
    m_tr, b_tr = np.polyfit(seg_ep, seg_tr, 1)
    fit_te = m_te * seg_ep + b_te
    fit_tr = m_tr * seg_ep + b_tr

    plt.figure(figsize=(10, 5))
    plt.plot(ep, tr_raw, linewidth=2.0, label='Train (raw)')
    plt.plot(ep, te_raw, linewidth=2.0, label='Test  (raw)')
    plt.plot(seg_ep, fit_te, ls='--', linewidth=2.2, label=f'Test OLS slope={m_te:.4g}')
    plt.plot(seg_ep, fit_tr, ls='--', linewidth=2.2, label=f'Train OLS slope={m_tr:.4g}')
    plt.axvline(anchor, ls=':', alpha=0.7, label=f'Anchor @ {anchor}')

    # x 軸只從錨點開始 & 裁掉左側
    if crop_left:
        plt.xlim(anchor, ep[-1])
    if xticks_from_anchor:
        total_span = int(ep[-1] - anchor) if ep[-1] > anchor else 1
        step = max(1, total_span // max_xticks)
        ticks = np.arange(anchor, ep[-1] + 1, step, dtype=int)
        if ticks[-1] != ep[-1]:
            ticks = np.append(ticks, ep[-1])
        plt.xticks(ticks)

    plt.title(f"Method B 整段趨勢線 — {metric_name}", fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_method_c_stepwise(df, metric_name, ylabel='Δ per epoch'):
    if df is None or df.empty:
        print(f"[{metric_name}] Method C 無資料可畫。")
        return
    x = df['epoch_to'].to_numpy()                 # 用右端點當 x
    y_test = df['Test_slope'].to_numpy(float)
    y_train = df['Train_slope'].to_numpy(float)

    # 加上各自 OLS 趨勢線（對 stepwise 斜率序列做回歸）
    m_te, b_te = np.polyfit(x, y_test, 1)
    m_tr, b_tr = np.polyfit(x, y_train, 1)
    fit_te = m_te * x + b_te
    fit_tr = m_tr * x + b_tr

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_test, marker='o', linewidth=1.8, label='Test stepwise slope')
    plt.plot(x, y_train, marker='o', linewidth=1.8, label='Train stepwise slope')
    plt.plot(x, fit_te, ls='--', linewidth=2.2, label=f'Test OLS slope={m_te:.4g}')
    plt.plot(x, fit_tr, ls='--', linewidth=2.2, label=f'Train OLS slope={m_tr:.4g}')
    plt.axhline(0.0, ls='--', alpha=0.6, linewidth=1.2)
    plt.title(f"Method C 接續斜率 — {metric_name}", fontsize=18)
    plt.xlabel('Epoch (to)', fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

# ---- 實際呼叫（只有 Method B / C）----
row_r2   = None if summary_overall is None or summary_overall.empty else summary_overall.iloc[0].to_dict()
row_rmse = None if summary_overall is None or summary_overall.empty else summary_overall.iloc[1].to_dict()
plot_method_b_segment(epochs, r2_tr,  r2_te,  row_r2,   metric_name='R² (RAW)',   ylabel='R²')
plot_method_b_segment(epochs, rmse_tr, rmse_te, row_rmse, metric_name='RMSE (RAW)', ylabel='RMSE')

plot_method_c_stepwise(r2_step_df,   metric_name='R² (RAW)',   ylabel='ΔR² per epoch')
plot_method_c_stepwise(rmse_step_df, metric_name='RMSE (RAW)', ylabel='ΔRMSE per epoch')
