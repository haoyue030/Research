import pandas as pd
import numpy as np

# 讀取固定點座標 Excel 檔案
fixed_points_file = "CrossSection_Reshaped_Extend_TWD97.xlsx"
df_fixed_points = pd.read_excel(fixed_points_file, header=0)

# 讀取 output.xlsx 檔案
file_path = r"D:\OneDrive\桌面\GetCoastalLine.V2\output_202410.xlsx"
df_output = pd.read_excel(file_path, header=0)

# 轉換 ID 為字串，確保匹配
df_output.columns = df_output.columns.astype(str)
df_fixed_points["ID"] = df_fixed_points["ID"].astype(str)

# 取得交集的 ID (確保兩個檔案中的 ID 一致)
common_ids = set(df_fixed_points["ID"]).intersection(df_output.columns)
numeric_ids = sorted([id_str for id_str in common_ids if id_str.isdigit()], key=int, reverse=True)

# 建立存儲距離的 DataFrame
distance_results = []

# 逐個處理每個影像
for index in range(0, len(df_output), 2):  # 每兩行為一組 (X, Y)
    filename = df_output.loc[index, "filename"]  # 取得影像名稱
    distance_data = {"filename": filename}  # 初始化存儲距離資料
    
    for section_id in numeric_ids:
        if section_id in df_fixed_points["ID"].values:
            fixed_x, fixed_y = df_fixed_points[df_fixed_points["ID"] == section_id][["X1", "Y1"]].values[0]

            # 確保座標數據完整 (避免 NaN)
            if section_id not in df_output.columns:
                continue  # 跳過無資料的 ID
            
            x_coord = df_output.loc[index, section_id]  # X 座標
            y_coord = df_output.loc[index + 1, section_id]  # Y 座標 (下一行)

            if pd.isna(x_coord) or pd.isna(y_coord):
                continue  # 如果座標缺失，則跳過該計算
            
            # 計算距離
            distance = np.sqrt((x_coord - fixed_x) ** 2 + (y_coord - fixed_y) ** 2)
            distance_data[section_id] = distance
    
    distance_results.append(distance_data)

# 轉為 DataFrame
result_df = pd.DataFrame(distance_results)

# 儲存結果到 Excel
output_result_file = f"Distance Results_10.xlsx"
result_df.to_excel(output_result_file, index=False)

print(f"計算完成，結果已儲存到 {output_result_file}")




# import pandas as pd

# # 讀取 Excel 檔案
# df = pd.read_excel('距離結果1.xlsx')

# # 計算 '距離' 這一列的平均值和標準偏差
# mean_value = df['距離'].mean()
# std_deviation = df['距離'].std()

# # 計算一倍標準偏差範圍
# lower_bound = mean_value - std_deviation
# upper_bound = mean_value + std_deviation

# # 輸出結果
# print("一倍標準偏差範圍: (", lower_bound, ",", upper_bound, ")")


# import pandas as pd
# import shutil
# import os

# # 讀取 Excel 檔案
# df = pd.read_excel('距離結果.xlsx')

# # 計算 '距離' 這一列的平均值和標準偏差
# mean_value = df['距離'].mean()
# std_deviation = df['距離'].std()

# # 計算一倍標準偏差範圍
# lower_bound = mean_value - 2*std_deviation
# upper_bound = mean_value + 2*std_deviation

# # 找出超出範圍的影像
# outliers = df[(df['距離'] < lower_bound) | (df['距離'] > upper_bound)]

# # 來源資料夾和目標資料夾
# source_folder = "D:\OneDrive\桌面\GetCoastalLine\output_images"  # 替換為實際影像存放路徑
# destination_folder = "D:\OneDrive\桌面\GetCoastalLine\篩選"  # 替換為存放超出範圍影像的目標資料夾

# # 確保目標資料夾存在，若不存在則建立
# os.makedirs(destination_folder, exist_ok=True)

# # 複製符合條件的影像到新資料夾
# for filename in outliers['filename']:
#     source_path = os.path.join(source_folder, filename)
#     destination_path = os.path.join(destination_folder, filename)
    
#     # 確保來源檔案存在後才複製
#     if os.path.exists(source_path):
#         shutil.copy(source_path, destination_path)
#         print(f"已複製: {filename}")
#     else:
#         print(f"檔案不存在: {filename}")

# print("篩選與複製完成！")
