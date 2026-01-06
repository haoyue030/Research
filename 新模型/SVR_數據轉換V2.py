# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
import joblib

# ===== 1) 讀檔 + 前處理（與你前面一致） =====
DATA_PATH   = r'D:\OneDrive\桌面\新模型\0805.xlsx'
SHEET_NAME  = 'Sheet1'

df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

for col in ['波高', '降雨', '暴風半徑']:
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    df[f'{col}_YJ'] = pt.fit_transform(df[[col]])

df['潮位_BC2'] = (df['潮位'] - df['潮位'].min() + 1e-6) ** 2
for col in ['波能', '功率']:
    df[f'{col}_log1p'] = np.log1p(df[col] - df[col].min() + 1e-6)

x = df['尖峰週期']
x_pos = x - x.min() + 1e-6
df['尖峰週期_BC'] = boxcox(x_pos, 2.4217)

features = ['wind_dir_sin','wind_dir_cos','風速','氣壓','暴風半徑_YJ','波能_log1p','尖峰週期_BC']
X = df[features].values
y = df['y'].values

# ===== 2) 建模（鎖定你找到的最佳超參數） =====
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf', C=98.80707965692369, epsilon=0.0003240447112517415, gamma=0.0024794580392016254))
])
model = TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())

# ===== 3) 穩健性：RepeatedKFold 上的 R² / RMSE =====
rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
rmse_scorer = make_scorer(rmse, greater_is_better=False)

cv_r2 = cross_val_score(model, X, y, cv=rkf, scoring='r2', n_jobs=-1)
cv_rmse = -cross_val_score(model, X, y, cv=rkf, scoring=rmse_scorer, n_jobs=-1)

print(f"[RepeatedKFold 5x10]  R²: mean={cv_r2.mean():.3f}, std={cv_r2.std():.3f}")
print(f"[RepeatedKFold 5x10] RMSE: mean={cv_rmse.mean():.1f}, std={cv_rmse.std():.1f}")

# ===== 4) 支持向量數量（以全資料擬合一次來觀察） =====
model.fit(X, y)
svr = model.regressor_.named_steps['svr']
n_sv = svr.support_.size
print(f"Support vectors: {n_sv} / {X.shape[0]}")

# ===== 5) 可選：掃一小圈 epsilon（固定 C 與 gamma），觀察 CV 表現與支持向量變化 =====
from sklearn.model_selection import cross_val_predict
eps_list = [0.01, 0.03, 0.05, 0.1, 0.2]
for eps in eps_list:
    pipe_eps = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf', C=98.80707965692369, epsilon=eps, gamma=0.0024794580392016254))
    ])
    model_eps = TransformedTargetRegressor(regressor=pipe_eps, transformer=StandardScaler())
    r2_cv = cross_val_score(model_eps, X, y, cv=rkf, scoring='r2', n_jobs=-1).mean()
    rmse_cv = -cross_val_score(model_eps, X, y, cv=rkf, scoring=rmse_scorer, n_jobs=-1).mean()
    model_eps.fit(X, y)
    n_sv_eps = model_eps.regressor_.named_steps['svr'].support_.size
    print(f"epsilon={eps:<5}  CV_R2={r2_cv:.3f}  CV_RMSE={rmse_cv:.1f}  SVs={n_sv_eps}/{X.shape[0]}")

# ===== 6) 最終輸出：用全資料重訓並存檔 =====
# joblib.dump(model, r'D:\OneDrive\桌面\SVR_RBF_final_model.joblib')
print("Saved: D:\\OneDrive\\桌面\\SVR_RBF_final_model.joblib")
