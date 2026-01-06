import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# ====== 設定路徑與參數 ======
excel_path = r"D:\OneDrive\桌面\GetCoastalLine.V2\output_coordinates.xlsx"
cross_section_file = "CrossSection_Reshaped_Extend_TWD97.xlsx"
filename_col = 'filename'
xul = 343628.589
yul = 2770952.645

# ====== 讀取資料 ======
df_output = pd.read_excel(excel_path, sheet_name="Sheet1")
df_cross_section = pd.read_excel(cross_section_file)

# ====== 處理時間欄位與篩選區間 ======
# 1) 從檔名擷取時間並轉成 datetime
df_output['datetime_str'] = df_output[filename_col].astype(str).str.extract(r'(\d{8}-\d{6})', expand=False)
df_output['datetime'] = pd.to_datetime(
    df_output['datetime_str'],
    format='%Y%m%d-%H%M%S',
    errors='coerce'  # 防呆：不合格式的給 NaT
)

# 2) 日期範圍（含整個 end_date 當天）
start_date = pd.to_datetime('2024-10-01')
end_date_exclusive = pd.to_datetime('2024-11-01') + pd.Timedelta(days=1)  # 10/31 之後的 00:00（不含）

mask_date = (df_output['datetime'] >= start_date) & (df_output['datetime'] < end_date_exclusive)

# 3) 每日時間範圍：排除 06:00 以前
mask_time = df_output['datetime'].dt.hour >= 5

# 4) 合併條件 & 丟掉 NaT
df_filtered = df_output[mask_date & mask_time & df_output['datetime'].notna()].copy()

# ====== 數值欄位（斷面距離編號）提取 ======
numeric_ids = df_output.select_dtypes(include=[np.number]).columns.tolist()

# ====== 畫圖開始 ======
fig, ax = plt.subplots(figsize=(6, 8))

# excluded_sections = list(range(540, 661, 20))  # 不畫 540～660
# break_sections = [680, 540]  # 中斷區段
excluded_sections = set()  # 保留變數但設為空集合
break_sections = []        # 目前不強制切斷，日後需要可再用
# ====== 用 colormap 畫多色岸線（切段 + 過濾 + 自動斷段） ======
num_lines = len(df_filtered) // 2
cmap      = plt.cm.get_cmap('tab20', num_lines)
colors    = cmap.colors
step      = 20  # 各斷面編號之間的預期距離

for idx in range(0, len(df_filtered), 2):
    color_idx = idx // 2
    color     = colors[color_idx]
    segments       = []  # 儲存所有段
    current_segment = []  # 正在收集的那一段
    prev_sec       = None

    # 由大到小走過所有斷面編號
    for section_id in sorted(numeric_ids, reverse=True):
        # 跳過 540～660
        if section_id in excluded_sections:
            continue
        # 欄位不存在或資料為 NaN 就跳過
        if section_id not in df_filtered.columns:
            continue
        x = df_filtered.iloc[idx  ][section_id]
        y = df_filtered.iloc[idx+1][section_id]
        if pd.isna(x) or pd.isna(y):
            continue

        # 前一個有效斷面跟這個斷面如果差距不是 step，就算斷線
        if prev_sec is not None and (prev_sec - section_id) != step:
            segments.append(current_segment)
            current_segment = []

        # 將這個點加到目前這段
        current_segment.append((x, y))
        prev_sec = section_id

    # 把最後一段也加進去
    if current_segment:
        segments.append(current_segment)

    # 繪圖：每一段都獨立畫，不同段就不連線
    for seg_idx, seg in enumerate(segments):
        if len(seg) < 2:
            continue
        xs, ys = zip(*seg)
        ax.plot(xs, ys,
                color=color,
                linewidth=1.5,
                label='岸線' if (idx == 0 and seg_idx == 0) else "")


# ========= 畫斷面線（也用原始座標） =========#
# 全部斷面都要畫（包含中斷區段）
for i, row in df_cross_section.iterrows():
    section = 740 - i*20  # 若你有 section 欄位，也可直接用 row['section']

    x1, y1 = row['X1'], row['Y1']
    x2, y2 = row['X2'], row['Y2']

    # 缺值就跳過，避免連線報錯
    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
        continue

    ax.plot([x1, x2], [y1, y2], color='gray', linewidth=1.0)
    ax.text(x2 + 5, y2, f"{int(section)}", fontsize=8,
            va='center', ha='left', clip_on=True)


# ========= 加上 Land / Sea Side ========= #
ax.text(343650, 2770300, "Land Side", fontsize=14, weight='bold')
ax.text(344050, 2770600, "Sea Side", fontsize=14, weight='bold')

# ========= 座標設定 ========= #
ax.set_xlim(343600, 344200)
ax.set_ylim(2770100, 2770900)


# 關閉 Y 軸科學記號
formatter = ScalarFormatter(useOffset=False, useMathText=False)
formatter.set_scientific(False)
ax.yaxis.set_major_formatter(formatter)

ax.set_title(f"2024年10月岸線")
# ax.legend(title="U-Net Coastal Line", loc="upper right")
plt.tight_layout()
plt.show()



