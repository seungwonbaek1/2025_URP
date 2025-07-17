import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# 경로 설정
split_dir = r"C:\Users\seung won\Desktop\대학\(2025) 여름방학 URP\data\토요타\augmented\forecast_500\split_0\train_augmented"
max_files_per_bin = 250

# 파일을 bin별로 정리
bin_files = defaultdict(list)
for file in sorted(os.listdir(split_dir)):
    if not file.endswith('.csv'):
        continue
    match = re.match(r"(\d+\.\d+_\d+\.\d+)_.*\.csv", file)
    if match:
        bin_key = match.group(1)
        bin_files[bin_key].append(file)

# bin별 그래프 시각화
for bin_key, files in bin_files.items():
    print(f"\n▶ SOH bin: {bin_key}, 파일 수: {len(files)}개")
    plt.figure(figsize=(10,6))

    for i, file in enumerate(files[:max_files_per_bin]):
        file_path = os.path.join(split_dir, file)
        df = pd.read_csv(file_path)
        if 'Cycle_Index' not in df.columns or 'Discharge_Capacity' not in df.columns:
            continue
        plt.plot(df['Cycle_Index'], df['Discharge_Capacity'], alpha=0.4)

    plt.title(f"SOH bin: [{bin_key.replace('_', ', ')}) - train_augmented")
    plt.xlabel("Cycle Index")
    plt.ylabel("Discharge Capacity")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
