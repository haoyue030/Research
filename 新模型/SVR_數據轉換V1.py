# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from scipy.stats import loguniform
import warnings
warnings.filterwarnings('ignore')

# =========================
# 🔧 參數（不做預篩；列舉所有合法組合；RBF 專屬）
# =========================
DATA_PATH   = r'D:\OneDrive\桌面\新模型\0805.xlsx'
SHEET_NAME  = 'Sheet1'
TEST_SIZE   = 0.30
RANDOM_SEED = 42

# 枚舉控制：以「單位」枚舉（兩組方向為成對單位，其餘單一單位）
# 若想真・全枚舉，請將 MAX_UNITS=None；若要控算量，可設 6~9
MAX_UNITS = None          # None = 使用 1..全部單位 的所有組合
EXCLUDE_FEATURES = []     # 要排除的特徵名稱（包含在成對單位中的任一成員則整個單位排除）
MUST_INCLUDE     = []     # 必含的特徵名稱（若屬於成對單位，則整個單位必含）

# 交叉驗證與評分
N_SPLITS = 5
SCORING  = 'neg_root_mean_squared_error'
N_ITER_RANDOM = 120  # 隨機粗搜次數（可降到 60 以加速）

# =========================
# 1) 讀資料 + 前處理
# =========================
df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

# Yeo–Johnson
for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

# 潮位平方
df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2

# 波能/功率 log1p（平移到 >=0）
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

# 尖峰週期 Box–Cox（lambda 2.4217）
x = df['尖峰週期']
x_pos = x - x.min() + 1e-6
df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

# 候選特徵（供索引與建模使用）
candidate_feats = [
    '風速', '氣壓',
    'wind_dir_sin','wind_dir_cos',
    'wave_dir_sin','wave_dir_cos',
    '波高_YJ', '降雨_YJ', '暴風半徑_YJ',
    '潮位_BC2',
    '波能_log1p', '功率_log1p',
    '尖峰週期_BC'
]

X_all = df[candidate_feats].values
y_all = df['y'].values

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_SEED
)

# =========================
# 2) 建立「單位」與工具
# =========================
# 將方向視為成對單位；其他皆單一單位
PAIR_UNITS = [
    ('wind_dir_sin','wind_dir_cos'),
    ('wave_dir_sin','wave_dir_cos'),
]
SINGLE_UNITS = [
    ('風速',), ('氣壓',),
    ('波高_YJ',), ('降雨_YJ',), ('暴風半徑_YJ',),
    ('潮位_BC2',),
    ('波能_log1p',), ('功率_log1p',),
    ('尖峰週期_BC',)
]
ALL_UNITS = PAIR_UNITS + SINGLE_UNITS  # 共 11 個「單位」

def feature_in_unit(u, f):
    return f in u

# 依 EXCLUDE_FEATURES 移除整個單位
filtered_units = []
for unit in ALL_UNITS:
    if any(f in EXCLUDE_FEATURES for f in unit):
        continue
    filtered_units.append(unit)

# 依 MUST_INCLUDE 標記必含單位
required_units = []
for unit in filtered_units:
    if any(f in MUST_INCLUDE for f in unit):
        required_units.append(unit)
# 去重
required_units = list(dict.fromkeys(required_units))

# 方便把「單位組合」展開為特徵清單
def flatten_features(unit_combo):
    feats = []
    for u in unit_combo:
        feats.extend(list(u))
    return feats

# RBF 模型與兩階段搜尋設定
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

base_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])
base_model = TransformedTargetRegressor(
    regressor=base_pipe,
    transformer=StandardScaler()
)

# 隨機粗搜空間（RBF）
param_distributions = [
    # 連續 gamma
    {
        'regressor__svr__kernel': ['rbf'],
        'regressor__svr__C': loguniform(1e-3, 1e3),
        'regressor__svr__epsilon': loguniform(1e-4, 2.0),
        'regressor__svr__gamma': loguniform(1e-4, 1e1)
    },
    # 內建 gamma：scale / auto
    {
        'regressor__svr__kernel': ['rbf'],
        'regressor__svr__C': loguniform(1e-3, 1e3),
        'regressor__svr__epsilon': loguniform(1e-4, 2.0),
        'regressor__svr__gamma': ['scale', 'auto']
    },
]

def stage2_grid_from_best_rbf(rnd_best_estimator):
    svr_step = rnd_best_estimator.regressor_.named_steps['svr']
    C0 = float(svr_step.C)
    eps0 = float(svr_step.epsilon)

    # gamma 基準
    g_used = getattr(svr_step, "_gamma", svr_step.gamma)
    if isinstance(g_used, str):
        try:
            nfeat = svr_step.n_features_in_
            g0 = 1.0 / max(1, nfeat)
        except Exception:
            g0 = 0.1
    else:
        g0 = float(g_used)

    C_grid = sorted({max(1e-6, C0 * f) for f in (0.2, 0.5, 1.0, 2.0, 5.0)})
    eps_grid = sorted({max(1e-6, eps0 * f) for f in (0.5, 0.8, 1.0, 1.2, 1.5)})
    gamma_grid_numeric = sorted({max(1e-8, g0 * f) for f in (0.25, 0.5, 1.0, 2.0, 4.0)})

    gamma_list = list(gamma_grid_numeric)
    if isinstance(svr_step.gamma, str):
        gamma_list += [svr_step.gamma]  # 保留 'scale'/'auto'

    return {
        'regressor__svr__kernel': ['rbf'],
        'regressor__svr__C': C_grid,
        'regressor__svr__epsilon': eps_grid,
        'regressor__svr__gamma': gamma_list
    }

# =========================
# 3) 列舉所有合法「單位組合」 + 兩階段搜尋（RBF）
# =========================
if MAX_UNITS is None:
    r_range = range(1, len(filtered_units) + 1)
else:
    r_range = range(1, min(MAX_UNITS, len(filtered_units)) + 1)

results = []
print("不做特徵預篩，將以『單位』列舉所有合法組合（方向成對）。")
print(f"可用單位數量：{len(filtered_units)}，必含單位數量：{len(required_units)}")
print("=" * 60)

for r in r_range:
    if r < len(required_units):
        continue
    print(f"▶ 處理單位組合大小 r = {r} ...")

    for unit_combo in combinations(filtered_units, r):
        # 必含單位約束
        if any(req not in unit_combo for req in required_units):
            continue

        # 展開為特徵清單與索引
        feats = flatten_features(unit_combo)
        idx = [candidate_feats.index(c) for c in feats]

        X_tr = X_train_all[:, idx]
        X_te = X_test_all[:, idx]
        y_tr = y_train_all
        y_te = y_test_all

        # Stage 1: 隨機粗搜（RBF）
        rnd = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=N_ITER_RANDOM,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            random_state=RANDOM_SEED,
            verbose=0
        )
        try:
            rnd.fit(X_tr, y_tr)
        except Exception as e:
            print(f"  ⚠ RandomizedSearch 失敗，單位組合 {unit_combo}: {e}")
            continue

        # Stage 2: 以最佳點為中心微調（RBF）
        grid = stage2_grid_from_best_rbf(rnd.best_estimator_)
        gcv = GridSearchCV(
            estimator=rnd.best_estimator_,
            param_grid=grid,
            cv=cv,
            scoring=SCORING,
            n_jobs=-1,
            verbose=0
        )
        try:
            gcv.fit(X_tr, y_tr)
        except Exception as e:
            print(f"  ⚠ GridSearch 失敗，單位組合 {unit_combo}: {e}")
            continue

        # 評估
        y_pred = gcv.best_estimator_.predict(X_te)
        mse_test = mean_squared_error(y_te, y_pred)
        rmse_test = float(np.sqrt(mse_test))
        r2_test = float(r2_score(y_te, y_pred))

        y_tr_pred = gcv.best_estimator_.predict(X_tr)
        r2_train = float(r2_score(y_tr, y_tr_pred))

        results.append({
            'Features': tuple(feats),
            'Num': len(feats),
            'Units': unit_combo,
            'Best_Kernel': 'rbf',
            'Best_Params': gcv.best_params_,
            'CV_RMSE': -gcv.best_score_,
            'R2_train': r2_train,
            'R2_test': r2_test,
            'MSE_test': mse_test,
            'RMSE_test': rmse_test
        })

# =========================
# 4) 結果整理
# =========================
print("\n搜尋完成，整理結果...")
res_df = pd.DataFrame(results)

if len(res_df) == 0:
    print("沒有找到符合條件的結果")
else:
    filtered = (
        res_df.query("R2_train > 0 and R2_test > 0.3")
              .sort_values('R2_test', ascending=False)
              .head(10)
    )
    if len(filtered) == 0:
        print("符合門檻的結果為空，改列出整體前 10：")
        filtered = res_df.sort_values('R2_test', ascending=False).head(10)

    # 友善列印
    def fmt_feats(tup):
        return '(' + ', '.join(f"'{f}'" for f in tup) + ')'
    filtered_print = filtered.copy()
    filtered_print['Features'] = filtered_print['Features'].apply(fmt_feats)

    print("\n" + "="*100)
    print("最佳結果 (Top 10):")
    print("="*100)
    print(filtered_print[['Features', 'Num', 'Best_Kernel', 'CV_RMSE', 'R2_train', 'R2_test', 'RMSE_test']].to_string(index=False))

    print("\n" + "="*50)
    print("最佳模型詳細參數:")
    print("="*50)
    best_row = filtered.iloc[0]
    print(f"特徵: {fmt_feats(best_row['Features'])}")
    print(f"核函數: {best_row['Best_Kernel']}")
    print(f"參數: {best_row['Best_Params']}")
    print(f"訓練集 R²: {best_row['R2_train']:.4f}")
    print(f"測試集 R²: {best_row['R2_test']:.4f}")
    print(f"測試集 RMSE: {best_row['RMSE_test']:.4f}")

print("\n處理完成！")
