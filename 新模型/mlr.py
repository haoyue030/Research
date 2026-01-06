import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只要有匯入就好，部分環境需要這行
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
# ------- 1. 建立示意用資料：y = β0 + β1*x1 + β2*x2 + 誤差 -------
np.random.seed(0)
n = 50  # 資料筆數

x1 = np.random.uniform(0, 10, n)   # 自變數1
x2 = np.random.uniform(0, 10, n)   # 自變數2

beta0, beta1, beta2 = 3, 1.5, -0.8  # 真實參數（示意）
noise = np.random.normal(0, 2, n)   # 加一點雜訊
y = beta0 + beta1 * x1 + beta2 * x2 + noise  # 應變數

# ------- 2. 用最小平方法估計迴歸係數 β_hat -------
# X = [1, x1, x2]
X = np.column_stack((np.ones(n), x1, x2))
beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
print("估計參數 β_hat =", beta_hat)

# ------- 3. 建立網格，畫出迴歸平面 -------
x1_grid, x2_grid = np.meshgrid(
    np.linspace(x1.min(), x1.max(), 20),
    np.linspace(x2.min(), x2.max(), 20)
)

y_grid = (beta_hat[0]
          + beta_hat[1] * x1_grid
          + beta_hat[2] * x2_grid)

# ------- 4. 繪圖：資料點 + 迴歸平面 -------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 資料點（散佈圖）
ax.scatter(x1, x2, y, alpha=0.8, label='觀測資料')

# 迴歸平面
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.4)

ax.set_xlabel('自變數 x1',fontsize = 16)
ax.set_ylabel('自變數 x2',fontsize = 16)
ax.set_zlabel('應變數 y',fontsize = 16)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)
ax.tick_params(axis='z', labelsize=14)
# ax.set_title('多元線性迴歸示意圖：資料點與迴歸平面')

# plt.legend(fontsize = 14)
plt.tight_layout()
plt.show()
