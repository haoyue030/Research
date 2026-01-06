from pathlib import Path
import shutil, csv

# === 路徑設定（請直接執行即可）===
P1 = Path(r"D:\OneDrive\桌面\2024影像\output_images")   # 路徑1：要比對的檔名清單所在
P2 = Path(r"D:\OneDrive\桌面\2024影像\12\output_images")      # 路徑2：實際來源（會遞迴搜尋）
P3 = Path(r"D:\OneDrive\桌面\2024影像\error")           # 路徑3：複製目標

# 影像副檔名白名單
EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif', '.webp'}

def is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in EXTS

P3.mkdir(parents=True, exist_ok=True)

# 1) 先把路徑2做索引（檔名 -> 多個候選完整路徑）
index = {}
for p in P2.rglob("*"):
    if is_img(p):
        key = p.name.lower()            # 以完整檔名比對（含副檔名）
        index.setdefault(key, []).append(p)

# 2) 逐一讀取路徑1的影像檔名，到索引中找來源，再複製到路徑3
p1_files = [p for p in P1.iterdir() if is_img(p)]
missing = []
copied = 0
duplicates_used_latest = 0

for f in p1_files:
    key = f.name.lower()
    candidates = index.get(key, [])
    if not candidates:
        missing.append(f.name)
        continue

    # 若在路徑2找到多個同名檔，選修改時間最新者
    src = max(candidates, key=lambda x: x.stat().st_mtime)
    if len(candidates) > 1:
        duplicates_used_latest += 1

    dst = P3 / f.name
    shutil.copy2(src, dst)
    copied += 1

# 3) 輸出簡易報告
report = P3 / "_copy_report.csv"
with report.open("w", newline="", encoding="utf-8-sig") as fh:
    w = csv.writer(fh)
    w.writerow(["total_in_path1", "copied", "not_found_in_path2", "duplicates_in_path2_used_latest"])
    w.writerow([len(p1_files), copied, len(missing), duplicates_used_latest])
    if missing:
        w.writerow([])
        w.writerow(["not_found_filenames"])
        for name in missing:
            w.writerow([name])

print(f"完成：已複製 {copied} 個檔案到 {P3}")
if missing:
    print(f"注意：{len(missing)} 個檔名在路徑2找不到，清單已寫入 {report}")
if duplicates_used_latest:
    print(f"提示：{duplicates_used_latest} 個檔名在路徑2出現多筆，已選取最新的那一筆。")
