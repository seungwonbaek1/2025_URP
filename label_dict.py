import os
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from collections import defaultdict

forecast_cycles = [500]
split_indices = [0, 1, 2]
splits = ["train_augmented", "trainval_augmented", "val", "test"]

AUGMENTED_DIR = r"C:\Users\seung won\Desktop\대학\2025 여름방학 URP\data\토요타\augmented"
LABEL_SAVE_DIR = r"D:\cnn_data\labels"
os.makedirs(LABEL_SAVE_DIR, exist_ok=True)

summary_stats = []

for cycle in forecast_cycles:
    for split_idx in split_indices:
        label_dict = {}
        soh_list = []
        bin_counts = defaultdict(int)

        print(f"\n📦 forecast_{cycle} split_{split_idx} 시작")

        for subset in splits:
            subset_dir = os.path.join(AUGMENTED_DIR, f"forecast_{cycle}", f"split_{split_idx}", subset)
            if not os.path.exists(subset_dir):
                print(f"⚠️ {subset_dir} 없음 → skip")
                continue

            csv_files = [f for f in os.listdir(subset_dir) if f.endswith(".csv")]
            print(f"  🔹 {subset}: {len(csv_files)}개")

            for fname in tqdm(csv_files, desc=f"[{subset}] 처리 중"):
                try:
                    fpath = os.path.join(subset_dir, fname)
                    df = pd.read_csv(fpath)
                    cap = df["Discharge_Capacity"].values

                    if len(cap) == 0:
                        continue

                    c0 = np.max(cap)
                    ck = cap[-1]
                    soh = round(ck / c0, 4) if c0 > 0 else 0.0

                    key = fname.replace(".csv", "")
                    label_dict[key] = soh
                    soh_list.append(soh)

                    # bin 분류
                    if soh < 0.8:
                        bin_counts["0.0~0.8"] += 1
                    elif soh < 0.85:
                        bin_counts["0.8~0.85"] += 1
                    elif soh < 0.9:
                        bin_counts["0.85~0.9"] += 1
                    elif soh < 0.95:
                        bin_counts["0.9~0.95"] += 1
                    else:
                        bin_counts["0.95~1.0"] += 1

                except Exception as e:
                    print(f"[ERROR] {fname} 오류: {e}")
                    continue

        # 저장
        save_path = os.path.join(LABEL_SAVE_DIR, f"forecast_{cycle}_split_{split_idx}_labels.json")
        with open(save_path, "w") as fp:
            json.dump(label_dict, fp, indent=2)

        # 요약 통계 저장
        if soh_list:
            stats = {
                "forecast_cycle": cycle,
                "split": split_idx,
                "count": len(soh_list),
                "mean": round(np.mean(soh_list), 4),
                "std": round(np.std(soh_list), 4),
                "min": round(np.min(soh_list), 4),
                "max": round(np.max(soh_list), 4),
                "0.0~0.8": bin_counts["0.0~0.8"],
                "0.8~0.85": bin_counts["0.8~0.85"],
                "0.85~0.9": bin_counts["0.85~0.9"],
                "0.9~0.95": bin_counts["0.9~0.95"],
                "0.95~1.0": bin_counts["0.95~1.0"]
            }
            summary_stats.append(stats)

        print(f"✅ 저장 완료 → {save_path} (총 {len(label_dict)}개)\n")

# 리포트 출력
df_report = pd.DataFrame(summary_stats)
print("\n📊 [최종 리포트]")
print(df_report.to_string(index=False))