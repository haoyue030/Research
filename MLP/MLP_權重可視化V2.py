# ======================================
# MLP + 學習曲線（RAW 版：峰值/斜率/繪圖）— 無 Method A
# + 權重可視化 & 連結權重法（平方版）
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

# # =========================
# # 權重可視化 & 特徵重要度（平方鏈版）
# # =========================
# def _safe_names(prefix, n):
#     return [f"{prefix}{i+1}" for i in range(n)]

# def plot_mlp_layer_weights(model, feature_names=None, target_names=None, show_values=False, figsize_per_layer=(7, 4)):
#     """逐層畫 sklearn.MLP 的權重熱圖（x=輸出節點, y=輸入節點）。"""
#     coefs = getattr(model, "coefs_", None)
#     if coefs is None:
#         raise ValueError("模型尚未訓練或不含 coefs_。請先 .fit() 後再可視化。")

#     n_layers = len(coefs)
#     layer_in_sizes  = [W.shape[0] for W in coefs]
#     layer_out_sizes = [W.shape[1] for W in coefs]

#     if feature_names is None:
#         feature_names = [f"x{i+1}" for i in range(layer_in_sizes[0])]
#     if target_names is None:
#         target_names = [f"y{i+1}" for i in range(layer_out_sizes[-1])]

#     for li, W in enumerate(coefs):
#         fig = plt.figure(figsize=figsize_per_layer)
#         ax = plt.gca()

#         vmax = float(np.max(np.abs(W))) if W.size else 1.0
#         im = ax.imshow(W, aspect='auto', cmap='bwr', vmin=-vmax, vmax=+vmax)

#         if li == 1:
#             ax.invert_yaxis()

#         cbar = plt.colorbar(im)
#         cbar.set_label("權重值", fontsize=16)
#         cbar.ax.tick_params(labelsize=12)

#         ylabels = feature_names if li == 0 else [f"h{li}_{i+1}" for i in range(W.shape[0])]
#         xlabels = target_names if li == n_layers - 1 else [f"h{li+1}_{i+1}" for i in range(W.shape[1])]

#         if len(ylabels) <= 30:
#             ax.set_yticks(range(len(ylabels)))
#             ax.set_yticklabels(ylabels, fontsize=14)
#         else:
#             ax.set_yticks([])

#         if len(xlabels) <= 30:
#             ax.set_xticks(range(len(xlabels)))
#             # ★ X 軸標籤轉 90 度
#             ax.set_xticklabels(xlabels, rotation=90, ha='right', fontsize=14)
#         else:
#             ax.set_xticks([])

#         ax.set_ylabel("輸入節點", fontsize=16)
#         ax.set_xlabel("輸出節點", fontsize=16)
#         ax.set_title(f"Layer {li+1} 權重  shape={W.shape}", fontsize=18)

#         if show_values and W.size <= 2000:
#             for in_i in range(W.shape[0]):
#                 for out_i in range(W.shape[1]):
#                     ax.text(out_i, in_i, f"{W[in_i, out_i]:.2f}",
#                             ha="center", va="center", fontsize=7)

#         plt.tight_layout()
#         plt.show()



# def _connection_weight_importance_squared(model, aggregate_outputs="equal"):
#     """連結權重法（平方版）：由輸出往回，以 W^2 連乘累積到輸入層。
#        回傳 (imp_sq, imp_signed)。imp_sq 用於排名；imp_signed 僅供粗略方向感參考。"""
#     coefs = model.coefs_
#     n_out = coefs[-1].shape[1]

#     # 輸出聚合（單輸出等同 1）
#     if aggregate_outputs == "equal" or aggregate_outputs is None:
#         v_sq = np.ones(n_out); v_sign = np.ones(n_out)
#     else:
#         v = np.asarray(aggregate_outputs).reshape(-1)
#         if v.size != n_out:
#             raise ValueError("aggregate_outputs 長度需等於輸出維度")
#         v_sq = np.abs(v)          # 聚合權重仍用正值
#         v_sign = v

#     # 平方版（排名用）：v_{l-1} = (W^{(l)})^2 @ v_l
#     for W in reversed(coefs):
#         v_sq = np.square(W) @ v_sq
#     imp_sq = v_sq.astype(float)

#     # 帶號版（參考）
#     vs = v_sign.copy()
#     for W in reversed(coefs):
#         vs = (W) @ vs
#     imp_signed = vs.astype(float)

#     return imp_sq, imp_signed

# def plot_mlp_input_importance(model, feature_names=None, aggregate_outputs="equal",
#                               top_k=None, normalize=True, figsize=(7, 5), title="MLP 輸入特徵重要度（連結權重法：平方版）"):
#     """整網路觀點的輸入特徵重要度長條圖（平方鏈）。"""
#     if not hasattr(model, "coefs_"):
#         raise ValueError("模型尚未訓練或不含 coefs_。")

#     n_in = model.coefs_[0].shape[0]
#     if feature_names is None:
#         feature_names = _safe_names("x", n_in)

#     imp_sq, imp_signed = _connection_weight_importance_squared(model, aggregate_outputs=aggregate_outputs)

#     # 正規化到和=1（顯示比例）
#     if normalize and imp_sq.sum() > 0:
#         imp_sq = imp_sq / imp_sq.sum()

#     order = np.argsort(imp_sq)[::-1]
#     if top_k is not None:
#         order = order[:top_k]

#     names_sorted = [feature_names[i] for i in order]
#     vals_sorted = imp_sq[order]

#     plt.figure(figsize=figsize)
#     plt.barh(range(len(order)), vals_sorted)
#     plt.yticks(range(len(order)), names_sorted)
#     plt.gca().invert_yaxis()
#     plt.xlabel("相對重要度（平方鏈，已正規化）")
#     plt.title(title)
#     plt.tight_layout(); plt.show()

#     return {
#         "feature_names_sorted": names_sorted,
#         "importance_sorted": vals_sorted,
#         "all_feature_names": feature_names,
#         "all_importance": imp_sq,
#         "signed_raw": imp_signed
#     }

# def plot_grouped_direction_pairs(feature_names, importance, pairs=None, title="方向特徵成對合併後的重要度（平方鏈）", figsize=(7,5)):
#     """將 direction 的 sin/cos 成對合併後再畫一次（避免拆開誤解）。"""
#     name_to_imp = {n: float(v) for n, v in zip(feature_names, importance)}
#     groups = []
#     if pairs is None:
#         pairs = [
#             (('wind_dir_sin', 'wind_dir_cos'), '風向 (sin+cos)'),
#             (('wave_dir_sin', 'wave_dir_cos'), '波向 (sin+cos)'),
#         ]
#     used = set()
#     for (a, b), label in pairs:
#         val = name_to_imp.get(a, 0.0) + name_to_imp.get(b, 0.0)
#         groups.append((label, val)); used.add(a); used.add(b)
#     # 其他未成對者照原名帶入
#     for n in feature_names:
#         if n not in used:
#             groups.append((n, name_to_imp.get(n, 0.0)))

#     groups.sort(key=lambda x: x[1], reverse=True)
#     labels = [g[0] for g in groups]
#     vals = [g[1] for g in groups]

#     plt.figure(figsize=figsize)
#     plt.barh(range(len(vals)), vals)
#     plt.yticks(range(len(vals)), labels)
#     plt.gca().invert_yaxis()
#     plt.xlabel("相對重要度（平方鏈，已正規化）")
#     plt.title(title)
#     plt.tight_layout(); plt.show()

# # ---------- 實際呼叫 ----------
# # 1) 各層權重熱圖
# plot_mlp_layer_weights(final_mlp, feature_names=FEATURE_COLS, target_names=['y'], show_values=False)

# # 2) 輸入特徵重要度（整網路，平方鏈）
# imp_res = plot_mlp_input_importance(final_mlp, feature_names=FEATURE_COLS,
#                                     aggregate_outputs="equal", top_k=None, normalize=True,
#                                     title="MLP 輸入特徵重要度（連結權重法：平方版）")

# # 3) 將方向 sin/cos 成對合併後再畫一次（平方鏈）
# plot_grouped_direction_pairs(
#     feature_names=imp_res["all_feature_names"],
#     importance=imp_res["all_importance"],
#     pairs=[ (('wind_dir_sin','wind_dir_cos'), '風向 (sin+cos)'),
#             (('wave_dir_sin','wave_dir_cos'), '波向 (sin+cos)') ],
#     title="方向特徵成對合併後的重要度（MLP 連結權重法：平方版）",
#     figsize=(7,5)
# )

# # （可選）把重要度存成 CSV：
# # import pandas as pd
# # pd.DataFrame({
# #     "feature": imp_res["feature_names_sorted"],
# #     "importance": imp_res["importance_sorted"]
# # }).to_csv("MLP_feature_importance_connection_weights_sq.csv", index=False, encoding="utf-8-sig")

# # ---------- 單一特徵（風速）逐路徑貢獻：平方鏈 ----------
# W1, W2, W3 = final_mlp.coefs_[0], final_mlp.coefs_[1], final_mlp.coefs_[2]  # shapes (8,20),(20,20),(20,1)

# feat_name = '風速'
# i = FEATURE_COLS.index(feat_name)  # 0-based index

# # 逐路徑精確計算（400 項），平方版：w1^2 * w2^2 * w3^2
# terms = []
# for j in range(W1.shape[1]):      # h1_j
#     for k in range(W2.shape[1]):  # h2_k
#         contrib = (W1[i, j]**2) * (W2[j, k]**2) * (W3[k, 0]**2)
#         terms.append((contrib, j, k))

# terms.sort(reverse=True, key=lambda x: x[0])
# imp_windspeed_sq = sum(t[0] for t in terms)  # 「風速」未正規化平方鏈分數

# # 同時計算「所有輸入」的平方鏈總和（向量化）
# v2 = np.square(W3).reshape(-1)   # (20,)
# v1 = np.square(W2) @ v2          # (20,)
# imp_all_sq = np.square(W1) @ v1  # (8,)
# imp_norm_sq = imp_windspeed_sq / (imp_all_sq.sum() + 1e-12)

# print(f"【平方鏈】特徵「{feat_name}」未正規化重要度 = {imp_windspeed_sq:.6e}")
# print(f"【平方鏈】特徵「{feat_name}」正規化比例     = {imp_norm_sq:.6%}")

# print("\n【平方鏈】Top-20 路徑貢獻（w1^2 * w2^2 * w3^2）：")
# for rank, (contrib, j, k) in enumerate(terms[:20], 1):
#     print(f"{rank:2d}. j=h1_{j+1:02d}, k=h2_{k+1:02d}, 貢獻={contrib:.6e}")
