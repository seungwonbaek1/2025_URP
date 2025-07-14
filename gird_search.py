# ✅ 최종 하이퍼파라미터 평균 리포트 스크립트 (professor 모델 기반)
import os
import json
from collections import defaultdict
import numpy as np
import pandas as pd

# ----------------------------
# 사용자 설정
# ----------------------------
DATA_DIR = r"D:\cnn_data"
forecast_cycle = 550
origin = 100
folds = [0, 1, 2]

transform_modes = [
    ("RP", "1ch"),
    ("GASF", "1ch"),
    ("GADF", "1ch"),
    ("GASF_GADF_2ch", "2ch"),
    ("RP_GAF_2stream", "2stream")
]

# transform_modes = [
#     ("RP", "1ch"),
#     ("GASF", "1ch"),
#     ("GADF", "1ch"),
#     ("GASF_GADF_2ch", "2ch"),
#     ("RP_GAF_2stream", "2stream")
# ]

# ----------------------------
# 통합 리포트용 결과 저장
# ----------------------------
final_report = []

print(f"\n📢 [시작] forecast_cycle={forecast_cycle}, origin={origin} 전체 모델 평균 성능 리포트\n")

for transform_type, mode in transform_modes:
    print(f"\n🔍 [{transform_type} - {mode}] 모델 처리 중...")

    results = defaultdict(list)
    found_any = False

    for fold in folds:
        json_path = os.path.join(
            DATA_DIR,
            f"forecast_{forecast_cycle}",
            f"fold_{fold}",
            f"origin_{origin}",
            transform_type,
            f"best_hparams_professor_{mode}.json"  # 🔄 수정됨
        )

        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
                key = (data["lr"], data["batch_size"])
                results[key].append((data["mae"], data["mape"], data["epoch"]))
                print(f"✅ fold {fold} 읽음 → lr={data['lr']}, bs={data['batch_size']}, mae={data['mae']}, mape={data['mape']}, epoch={data['epoch']}")
                found_any = True
        else:
            print(f"❌ fold {fold} → 파일 없음: {json_path}")

    if not found_any:
        print("🚫 유효한 json 파일이 없음 → 이 모델은 스킵됩니다.")
        continue

    # 평균 계산
    avg_results = []
    for (lr, bs), scores in results.items():
        maes, mapes, epochs = zip(*scores)
        avg_results.append({
            "lr": lr,
            "batch_size": bs,
            "mae_mean": round(np.mean(maes), 5),
            "mape_mean": round(np.mean(mapes), 5),
            "epoch_mean": round(np.mean(epochs), 1),
            "count": len(scores)
        })

    df = pd.DataFrame(avg_results)
    df = df.sort_values(by="mae_mean").reset_index(drop=True)

    best = df.iloc[0]
    print(f"\n🌝 [결과] {transform_type} 모델 최적 하이퍼파라미터:")
    print(f" → learning rate: {best.lr}")
    print(f" → batch size   : {int(best.batch_size)}")
    print(f" → avg MAE      : {best.mae_mean}")
    print(f" → avg MAPE     : {best.mape_mean}")
    print(f" → avg Epoch    : {best.epoch_mean}\n")

    # 리포트 저장
    final_report.append({
        "model": transform_type,
        "mode": mode,
        "lr": best.lr,
        "batch_size": int(best.batch_size),
        "avg_mae": best.mae_mean,
        "avg_mape": best.mape_mean,
        "avg_epoch": best.epoch_mean
    })

    # 🔐 저장: 향후 test.py에서 직접 로딩 가능하게 저장
    save_json = os.path.join(
        DATA_DIR,
        f"forecast_{forecast_cycle}",
        f"origin_{origin}",
        f"best_hparam_professor_{transform_type}_{mode}.json"
    )
    os.makedirs(os.path.dirname(save_json), exist_ok=True)
    with open(save_json, "w") as f:
        json.dump({
            "lr": best.lr,
            "batch_size": int(best.batch_size),
            "epoch": best.epoch_mean
        }, f, indent=2)
    print(f"💾 저장 완료 → {save_json}")

# ----------------------------
# 최종 통합 리포트 출력
# ----------------------------
print("\n==============================")
print("📊 전체 모델 최종 성능 리포트")
print("==============================\n")

report_df = pd.DataFrame(final_report)
report_df = report_df.sort_values(by="avg_mae").reset_index(drop=True)

print(report_df.to_string(index=False))

# 최상위 베스트 모델 요약
if not report_df.empty:
    best_row = report_df.iloc[0]
    print(f"\n🌿 최종 Best Model: {best_row.model} ({best_row.mode})")
    print(f" → learning rate: {best_row.lr}, batch size: {best_row.batch_size}")
    print(f" → avg MAE: {best_row.avg_mae}, avg MAPE: {best_row.avg_mape}, avg Epoch: {best_row.avg_epoch}")
else:
    print("⚠️ 어떤 모델도 유효한 결과를 찾지 못했습니다.")
