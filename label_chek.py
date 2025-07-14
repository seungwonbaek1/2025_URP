import os
import json
import numpy as np
import pandas as pd
from collections import Counter

LABEL_DIR = r"D:\cnn_data\labels"
forecast_cycles = [500]
split_indices = [0, 1, 2]

# SOH bin 기준
bin_ranges = [(0.0, 0.8), (0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
bin_labels = ["0.0~0.8", "0.8~0.85", "0.85~0.9", "0.9~0.95", "0.95~1.0"]

summary = []

for cycle in forecast_cycles:
    for split in split_indices:
        label_path = os.path.join(LABEL_DIR, f"forecast_{cycle}_split_{split}_labels.json")
        if not os.path.exists(label_path):
            print(f"[경고] 없음: {label_path}")
            continue
        
        with open(label_path) as f:
            label_dict = json.load(f)
        
        sohs = np.array(list(label_dict.values()))
        bin_counts = Counter()

        for soh in sohs:
            for (low, high), label in zip(bin_ranges, bin_labels):
                if low <= soh < high or (soh == 1.0 and high == 1.0):
                    bin_counts[label] += 1
                    break
        
        summary.append({
            "forecast_cycle": cycle,
            "split": split,
            "count": len(sohs),
            "mean": round(np.mean(sohs), 4),
            "std": round(np.std(sohs), 4),
            "min": round(np.min(sohs), 4),
            "max": round(np.max(sohs), 4),
            **{label: bin_counts.get(label, 0) for label in bin_labels}
        })

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
