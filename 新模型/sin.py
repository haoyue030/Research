import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
angle = 101  # 度（羅經角：0°北，順時針）
rad = np.deg2rad(angle)

# 關鍵：x = sin(θ), y = cos(θ)
x = np.sin(rad)   # ≈  0.9816（向東）
y = np.cos(rad)   # ≈ -0.1908（向南）

plt.figure(figsize=(6,6))
ax = plt.gca()
ax.add_patch(plt.Circle((0,0), 1, fill=False, linestyle='--'))

plt.arrow(0, 0, x, y, head_width=0.05, head_length=0.1, linewidth=2)

# 標註往下移一些
plt.text(x - 0.25, y - 0.18,
         f"角度: {angle}°\nsin={x:.3f}（東西）\ncos={y:.3f}（南北）",
         ha='center', va='center', fontsize=14,
         bbox=dict(facecolor='white', alpha=0.6))

# 參考方位
plt.text(0, 1.15, 'N', ha='center', va='center', fontsize=12)
plt.text(1.15, 0, 'E', ha='center', va='center', fontsize=12)
plt.text(0, -1.15,'S', ha='center', va='center', fontsize=12)
plt.text(-1.15,0, 'W', ha='center', va='center', fontsize=12)

plt.axhline(0, color='black', linewidth=0.8)
plt.axvline(0, color='black', linewidth=0.8)
plt.title("波向 101° 的 sin/cos 向量（0°北，順時針）", fontsize=14)
plt.xlabel("sin(θ) → 東西分量（東+ 西-）", fontsize=14)
plt.ylabel("cos(θ) → 南北分量（北+ 南-）", fontsize=14)
plt.xticks(fontsize=14); plt.yticks(fontsize=14)
ax.set_aspect('equal'); plt.grid(); plt.show()
