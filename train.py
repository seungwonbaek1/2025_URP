import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BatterySOHDataset
from model import CNN_Professor, TwoStreamCNN_Professor
from tqdm import tqdm
import numpy as np
from itertools import product
import pandas as pd
import os
import json

# -------------------------
# 하이퍼파라미터 설정
# -------------------------
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [16, 32]
epochs = 50  # Early stopping 사용하므로 크게 설정
patience = 3

# -------------------------
# 경로 설정
# -------------------------
DATA_DIR = r"D:\cnn_data"
LABEL_DIR = os.path.join(DATA_DIR, "labels")

forecast_cycles = [500]
split_indices = [0]
origins = [100, 200, 300]

transform_modes = [
    ("RP", "1ch"),
    ("GASF_GADF_2ch", "2ch"),
    ("RP_GAF_2stream", "2stream")
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_report = []

# -------------------------
# 메인 반복
# -------------------------
for cycle in forecast_cycles:
    for split_idx in split_indices:
        label_path = os.path.join(LABEL_DIR, f"forecast_{cycle}_split_{split_idx}_labels.json")
        with open(label_path) as f:
            label_dict = json.load(f)

        for origin in origins:
            for transform_type, mode in transform_modes:
                print(f"\n==============================")
                print(f"🔍 [forecast_{cycle}] split={split_idx} origin={origin} model={transform_type}")
                print(f"==============================")

                best_result = {"mae": float('inf'), "mape": float('inf'), "epoch": -1, "lr": None, "batch_size": None}
                best_model_state = None

                for lr, batch_size in product(learning_rates, batch_sizes):
                    print(f"\n🚀 [TRY] lr={lr}, batch_size={batch_size}")

                    train_dir = os.path.join(DATA_DIR, f"forecast_{cycle}", f"split_{split_idx}", f"origin_{origin}", transform_type, "train_augmented")
                    val_dir = os.path.join(DATA_DIR, f"forecast_{cycle}", f"split_{split_idx}", f"origin_{origin}", transform_type, "val")

                    train_files = [os.path.join(train_dir, f) for f in os.listdir(train_dir) if f.endswith(".npy") or f.endswith(".npz")]
                    val_files = [os.path.join(val_dir, f) for f in os.listdir(val_dir) if f.endswith(".npy") or f.endswith(".npz")]

                    dataset_train = BatterySOHDataset(train_files, label_dict, mode=mode)
                    dataset_val = BatterySOHDataset(val_files, label_dict, mode=mode)

                    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

                    if mode in ["1ch", "2ch"]:
                        model = CNN_Professor(in_channels=1 if mode == "1ch" else 2)
                    elif mode == "2stream":
                        model = TwoStreamCNN_Professor()

                    model.to(device)
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    best_mae = float('inf')
                    best_mape = float('inf')
                    best_epoch = -1
                    best_tmp_state = None
                    no_improve_count = 0

                    for epoch in range(epochs):
                        model.train()
                        for x, y in tqdm(loader_train, desc=f"[{transform_type}] Epoch {epoch+1}/{epochs}"):
                            optimizer.zero_grad()
                            if mode == "2stream":
                                x1, x2 = x
                                x1, x2 = x1.to(device), x2.to(device)
                                y = y.to(device).view(-1)
                                pred = model((x1, x2)).view(-1)
                            else:
                                x = x.to(device)
                                y = y.to(device).view(-1)
                                pred = model(x).view(-1)

                            loss = criterion(pred, y)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        mae_list = []
                        mape_list = []
                        with torch.no_grad():
                            for x, y in loader_val:
                                if mode == "2stream":
                                    x1, x2 = x
                                    x1, x2 = x1.to(device), x2.to(device)
                                    y = y.to(device).view(-1)
                                    pred = model((x1, x2)).view(-1)
                                else:
                                    x = x.to(device)
                                    y = y.to(device).view(-1)
                                    pred = model(x).view(-1)

                                mae = torch.mean(torch.abs(pred - y)).item()
                                mape = torch.mean(torch.abs((pred - y) / (y + 1e-8))).item()
                                mae_list.append(mae)
                                mape_list.append(mape)

                        val_mae = np.mean(mae_list)
                        val_mape = np.mean(mape_list)
                        print(f"📈 Epoch {epoch+1:2d} → val_mae={val_mae:.5f}, val_mape={val_mape:.5f}")

                        if val_mae < best_mae:
                            best_mae = val_mae
                            best_mape = val_mape
                            best_epoch = epoch + 1
                            best_tmp_state = model.state_dict()
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                            if no_improve_count >= patience:
                                print(f"⏹️ Early stopping triggered at epoch {epoch+1}")
                                break

                    print(f"\n✅ 조합 결과 (lr={lr}, batch_size={batch_size})")
                    print(f"   → best_epoch={best_epoch}, best_mae={best_mae:.5f}, best_mape={best_mape:.5f}")

                    if best_mae < best_result["mae"]:
                        best_result = {
                            "mae": round(best_mae, 5),
                            "mape": round(best_mape, 5),
                            "epoch": best_epoch,
                            "lr": lr,
                            "batch_size": batch_size
                        }
                        best_model_state = best_tmp_state

                result_path = os.path.join(DATA_DIR, f"forecast_{cycle}", f"split_{split_idx}", f"origin_{origin}", transform_type, f"best_hparams_professor_{mode}.json")
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                with open(result_path, "w") as f:
                    json.dump(best_result, f, indent=2)

                if best_model_state:
                    model.load_state_dict(best_model_state)
                    save_path = os.path.join(DATA_DIR, f"forecast_{cycle}", f"split_{split_idx}", f"origin_{origin}", transform_type, f"best_model_professor_{mode}.pth")
                    torch.save(model.state_dict(), save_path)

                print(f"\n🌝 [탐색 완료] model={transform_type}, mode={mode}")
                print(f" → best_epoch: {best_result['epoch']}")
                print(f" → best_mae  : {best_result['mae']}")
                print(f" → best_mape : {best_result['mape']}")
                print(f" → best_lr   : {best_result['lr']}")
                print(f" → best_batch_size: {best_result['batch_size']}")
                print(f" → 저장 경로 : {result_path}")

                final_report.append({
                    "forecast_cycle": cycle,
                    "model": transform_type,
                    "mode": mode,
                    "split": split_idx,
                    "origin": origin,
                    "lr": best_result["lr"],
                    "batch_size": best_result["batch_size"],
                    "epoch": best_result["epoch"],
                    "mae": best_result["mae"],
                    "mape": best_result["mape"]
                })

# -------------------------
# 최종 리포트 출력
# -------------------------
print("\n\n==============================")
print("📊 전체 모델 최종 탐색 리포트")
print("==============================\n")

df_report = pd.DataFrame(final_report)
df_report = df_report.sort_values(by=["origin", "mae"]).reset_index(drop=True)
print(df_report.to_string(index=False))

# 🔍 origin 기준 하위 리포트 출력
for ori in sorted(df_report["origin"].unique()):
    df_sub = df_report[df_report["origin"] == ori].sort_values(by="mae").reset_index(drop=True)
    print(f"\n🔹 origin = {ori}")
    print(df_sub.to_string(index=False))

    best = df_sub.iloc[0]
    print(f"\n🌟 origin {ori} Best:")
    print(f" → model: {best.model} ({best.mode}), split={best.split}")
    print(f" → lr={best.lr}, batch_size={best.batch_size}, epoch={best.epoch}")
    print(f" → mae={best.mae}, mape={best.mape}")

# 🔢 epoch 통계 정보 출력
print("\n\n==============================")
print("⏱️ Epoch Statistics")
print("==============================")
print(f"Min epoch   : {df_report['epoch'].min()}")
print(f"Max epoch   : {df_report['epoch'].max()}")
print(f"Mean epoch  : {df_report['epoch'].mean():.2f}")
print(f"Std  epoch  : {df_report['epoch'].std():.2f}")

print("\n\n🔍 [요약] (origin, split, forecast_cycle, model) 별 최적 하이퍼파리터")
for row in df_report.itertuples():
    key = (row.origin, row.split, row.forecast_cycle, row.model)
    print(f"{key} : lr = {row.lr}, bs = {row.batch_size}")

if df_report.empty:
    print("⚠️ 저장된 결과가 없습니다.")
