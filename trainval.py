import os
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import BatterySOHDataset
from model import CNN_Professor, TwoStreamCNN_Professor
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

# -------------------------
# 경로 설정
# -------------------------
DATA_DIR = r"D:\cnn_data"
LABEL_DIR = os.path.join(DATA_DIR, "labels")
CAM_SAVE_DIR = os.path.join(DATA_DIR, "cam")
os.makedirs(CAM_SAVE_DIR, exist_ok=True)

forecast_cycles = [500] #수정사항
splits = [0]
origins = [100]
transform_modes = [
    ("RP_GAF_2stream", "2stream")
]

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# CAM 생성 함수
# -------------------------
def save_cam(feature_map, weights, label, save_path, origin_length=None):
    if weights.dim() > 1:
        weights = weights.squeeze()
    cam = torch.einsum('c,chw->hw', weights, feature_map)
    cam = cam.detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cv2.resize(cam, (224, 224))

    plt.figure(figsize=(4, 4))
    plt.imshow(cam, cmap='jet', alpha=0.5)
    plt.title(f"Predicted SOH: {label:.4f}")
    plt.xlabel("Cycle index")
    plt.ylabel("Cycle index")

    if origin_length:
        ticks = np.linspace(0, 224, 5)
        tick_labels = [f"{int(origin_length * r)}" for r in [0.0, 0.25, 0.5, 0.75, 1.0]]
        plt.xticks(ticks, labels=tick_labels)
        plt.yticks(ticks, labels=tick_labels)
    else:
        plt.xticks(np.linspace(0, 224, 5), labels=["0", "25%", "50%", "75%", "100%"])
        plt.yticks(np.linspace(0, 224, 5), labels=["0", "25%", "50%", "75%", "100%"])

    plt.grid(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# -------------------------
# 학습/평가 루프
# -------------------------
for cycle in forecast_cycles:
    for split in splits:
        with open(os.path.join(LABEL_DIR, f"forecast_{cycle}_split_{split}_labels.json")) as f:
            label_dict = json.load(f)

        for origin in origins:
            for transform_type, mode in transform_modes:
                print(f"\n========== \U0001F680 split {split} | origin {origin} | type {transform_type} ==========")

                hparam_path = os.path.join(
                    DATA_DIR, f"forecast_{cycle}", f"split_{split}",
                    f"origin_{origin}", transform_type,
                    f"best_hparams_professor_{mode}.json"
                )
                if not os.path.exists(hparam_path):
                    print(f"\u274c 하이퍼파라미터 파일 없음: {hparam_path}")
                    continue

                with open(hparam_path) as hp:
                    best_hparam = json.load(hp)
                lr = best_hparam["lr"]
                batch_size = best_hparam["batch_size"]
                epochs = 15

                print(f"✅ Best hyperparameters: lr={lr}, batch_size={batch_size}, epochs=15 (fixed)")

                trainval_dir = os.path.join(
                    DATA_DIR, f"forecast_{cycle}", f"split_{split}",
                    f"origin_{origin}", transform_type, "trainval_augmented"
                )
                test_dir = os.path.join(
                    DATA_DIR, f"forecast_{cycle}", f"split_{split}",
                    f"origin_{origin}", transform_type, "test"
                )

                trainval_files = [os.path.join(trainval_dir, f) for f in os.listdir(trainval_dir) if f.endswith(".npy") or f.endswith(".npz")]
                test_files = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if f.endswith(".npy") or f.endswith(".npz")]

                dataset_train = BatterySOHDataset(trainval_files, label_dict, mode=mode)
                dataset_test = BatterySOHDataset(test_files, label_dict, mode=mode)

                loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
                loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

                if mode in ["1ch", "2ch"]:
                    model = CNN_Professor(in_channels=1 if mode == "1ch" else 2)
                elif mode == "2stream":
                    model = TwoStreamCNN_Professor()
                model.to(device)

                criterion = nn.MSELoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)

                final_mae_list = []
                final_mape_list = []

                # Epoch 단위 학습 + 평가
                for epoch in range(epochs):
                    print(f"\n\U0001F501 Epoch {epoch+1}/{epochs} 학습 시작...")
                    model.train()
                    loop = tqdm(loader_train, desc=f"[Epoch {epoch+1}/{epochs}]", ncols=100)
                    for x, y in loop:
                        optimizer.zero_grad()
                        if mode == "2stream":
                            x1, x2 = x
                            x1, x2 = x1.to(device), x2.to(device)
                            y = y.to(device)
                            pred = model((x1, x2)).squeeze()
                        else:
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x).squeeze()
                        loss = criterion(pred, y)
                        loss.backward()
                        optimizer.step()

                    # 매 epoch 후 평가
                    model.eval()
                    mae_list = []
                    mape_list = []
                    with torch.no_grad():
                        for x, y in loader_test:
                            if mode == "2stream":
                                x1, x2 = x
                                x1, x2 = x1.to(device), x2.to(device)
                                y = y.to(device)
                                pred = model((x1, x2)).squeeze()
                            else:
                                x = x.to(device)
                                y = y.to(device)
                                pred = model(x).squeeze()

                            mae = torch.mean(torch.abs(pred - y)).item()
                            mape = torch.mean(torch.abs((pred - y) / (y + 1e-8))).item()
                            mae_list.append(mae)
                            mape_list.append(mape)

                    avg_mae = np.mean(mae_list)
                    avg_mape = np.mean(mape_list)
                    final_mae_list = mae_list
                    final_mape_list = mape_list
                    print(f"\U0001F4CA [Epoch {epoch+1}] Test MAE: {avg_mae:.5f} | Test MAPE: {avg_mape:.5f}")

                # 최종 평가 리포트 출력
                print(f"\n✅ 최종 평가 결과 (after {epochs} epochs):")
                print(f"   - Test MAE:  {np.mean(final_mae_list):.5f}")
                print(f"   - Test MAPE: {np.mean(final_mape_list):.5f}")
                print(f"   - Total test samples: {len(final_mae_list)}")

                # CAM 저장은 마지막에만
                activation = {}
                def hook_fn(module, input, output):
                    activation['feature'] = output.squeeze(0).detach()

                if mode == "2stream":
                    model.stream_rp.conv2.register_forward_hook(
                        lambda m, i, o: activation.update({'feature_rp': o.squeeze(0).detach()})
                    )
                    model.stream_gaf.conv2.register_forward_hook(
                        lambda m, i, o: activation.update({'feature_gaf': o.squeeze(0).detach()})
                    )
                    fc_weights_rp = model.stream_rp.fc.weight.squeeze().detach().cpu()
                    fc_weights_gaf = model.stream_gaf.fc.weight.squeeze().detach().cpu()
                else:
                    model.conv2.register_forward_hook(hook_fn)
                    fc_weights = model.fc.weight.squeeze().detach().cpu()

                save_root = os.path.join(CAM_SAVE_DIR, f"forecast_{cycle}", f"split_{split}",
                                         f"origin_{origin}", transform_type)
                os.makedirs(save_root, exist_ok=True)

                model.eval()
                with torch.no_grad():
                    for i, (x, y) in enumerate(tqdm(loader_test, desc="\U0001F4F8 Saving CAM", ncols=100)):
                        if mode == "2stream":
                            x1, x2 = x
                            x1, x2 = x1.to(device), x2.to(device)
                            y = y.to(device)
                            pred = model((x1, x2)).squeeze()

                            fmap_rp = activation['feature_rp']
                            fmap_gaf = activation['feature_gaf']
                            file_name = os.path.basename(test_files[i]).replace(".npy", "").replace(".npz", "")
                            save_path_rp = os.path.join(save_root, f"{file_name}_rp.png")
                            save_path_gaf = os.path.join(save_root, f"{file_name}_gaf.png")
                            print(f"[DEBUG] Saving CAMs for: {file_name}")
                            print("  - RP feature map shape:", fmap_rp.shape)
                            print("  - GAF feature map shape:", fmap_gaf.shape)
                            save_cam(fmap_rp, fc_weights_rp, pred.item(), save_path_rp, origin)
                            save_cam(fmap_gaf, fc_weights_gaf, pred.item(), save_path_gaf, origin)
                        else:
                            x = x.to(device)
                            y = y.to(device)
                            pred = model(x).squeeze()

                            fmap = activation['feature']
                            file_name = os.path.basename(test_files[i]).replace(".npy", "").replace(".npz", "")
                            save_path = os.path.join(save_root, f"{file_name}.png")
                            save_cam(fmap, fc_weights, pred.item(), save_path, origin)

print("\n\U0001F389 전체 학습 + Epoch별 성능 평가 + CAM 저장 완료!")
