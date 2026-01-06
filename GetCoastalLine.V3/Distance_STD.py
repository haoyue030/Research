'''
篩選出範圍以外的異常影像，並複製到新資料夾
'''
import pandas as pd
import numpy as np
import os
import shutil

# ===== 路徑設定 =====
file_path = r"D:\OneDrive\桌面\GetCoastalLine.V2\Distance Results_10.xlsx"
source_folder = r"D:\OneDrive\桌面\GetCoastalLine.V2\20251014\output_images"
destination_folder = r"D:\OneDrive\桌面\GetCoastalLine.V2\Out_errorimage"
coor_path = r"D:\OneDrive\桌面\GetCoastalLine.V2\20251014\output_202410.xlsx"
output_distance = r"output_distance.xlsx"
output_coor =  r"output_coordinates.xlsx"
# ===== 讀取資料 =====
df = pd.read_excel(file_path, sheet_name='Sheet1')

# 排除不使用的斷面(篩選時使用)
excluded_sections = [str(x) for x in range(540, 661, 20)]
section_cols = [col for col in df.columns if col != 'filename' and str(col) not in excluded_sections]
##選取全部斷面
# section_cols = [col for col in df.columns if col != 'filename']

# ===== 計算 標準偏差 並找出異常 filename =====
stats = []
outlier_filenames = set()

for col in section_cols:
    mean = df[col].mean()
    std = df[col].std()
    max = df[col].max()
    min = df[col].min()
    # 動態設定倍數
    if std > 8:
        multiplier = 5
    elif std > 5:
        multiplier = 4

    lower = mean - multiplier * std
    upper = mean + multiplier * std

    stats.append({
        'section': col,
        'mean': mean,
        'std': std,
        'min':min,
        'max':max,
        'multiplier': multiplier,
        'lower_bound': lower,
        'upper_bound': upper
    })

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    outlier_filenames.update(outliers['filename'])

# 輸出統計表
stats_df = pd.DataFrame(stats)
print(stats_df)

# ===== 複製異常影像檔案到新資料夾 =====
os.makedirs(destination_folder, exist_ok=True)

for filename in outlier_filenames:
    source_path = os.path.join(source_folder, filename)
    destination_path = os.path.join(destination_folder, filename)
    
    if os.path.exists(source_path):
        shutil.copy2(source_path, destination_path)
        print(f"✅ 已複製：{filename}")
    else:
        print(f"⚠️ 找不到檔案：{filename}")

# ====================================篩選後重新計算標準偏差=========================================
df_normal = df[~df['filename'].isin(outlier_filenames)]
post_stats = []
for col in section_cols:
    mean = df_normal[col].mean()
    std = df_normal[col].std()
    min_val = df_normal[col].min()
    max_val = df_normal[col].max()

    post_stats.append({
        'section': col,
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val
    })
print(pd.DataFrame(post_stats))

#============================= 篩選後的distance數據複製到新excel==============================
valid_filenames = df[~df['filename'].isin(outlier_filenames)]['filename']
filtered_df = df[df['filename'].isin(valid_filenames)]

# ===== 寫入新 Excel 檔案 =====
filtered_df.to_excel(output_distance, index=False)
print(f"\n✅ 已儲存篩選後的資料到：{output_distance}")

#============================= 篩選後的croo數據==============================================
df_coord_table = pd.read_excel(coor_path)

# ===== 篩選出符合的列 =====
df_coord_filtered = df_coord_table[df_coord_table['filename'].isin(valid_filenames)]

# ===== 輸出新的 Excel 檔 =====
df_coord_filtered.to_excel(output_coor, index=False)
print(f"✅ 已儲存符合篩選的座標列到：{output_coor}")
