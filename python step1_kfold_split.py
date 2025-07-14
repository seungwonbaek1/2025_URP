import os
import numpy as np
import pandas as pd
from pyts.image import RecurrencePlot, GramianAngularField
from tqdm import tqdm
import gc
import traceback

# ============================
# 변환 함수
# ============================
def transform_rp(x):
    rp = RecurrencePlot(threshold='point', percentage=20)
    return rp.fit_transform(x.reshape(1, -1))[0]

def transform_gaf(x, method="summation"):
    gaf = GramianAngularField(method=method)
    return gaf.fit_transform(x.reshape(1, -1))[0]

# ============================
# 파라미터
# ============================
forecast_cycles = [550]
folds = [0,1,2]
origins = [100, 150, 200, 250, 300]
splits = ["train_augmented", "trainval_augmented", "val", "test"]

AUGMENTED_DIR = r"C:\Users\seung won\Desktop\대학\2025 여름방학 URP\data\토요타\augmented"
SAVE_DIR = r"D:\cnn_data"

# ============================
# 메인
# ============================
for cycle in forecast_cycles:
    for fold in folds:
        print(f"\n========== forecast_{cycle} fold_{fold} 시작 ==========")

        for origin in origins:
            for split in splits:
                try:
                    src_dir = os.path.join(
                        AUGMENTED_DIR,
                        f"forecast_{cycle}",
                        f"fold_{fold}",
                        split
                    )
                    file_list = [f for f in os.listdir(src_dir) if f.endswith(".csv")]
                    print(f"\n[INFO] origin {origin} split {split}: {len(file_list)}개 처리 예정")

                    file_count = 0

                    for file in tqdm(file_list, desc=f"{cycle}-{fold}-{origin}-{split}"):

                        try:
                            df = pd.read_csv(os.path.join(src_dir, file))
                            y = df["Discharge_Capacity"].values

                            if len(y) < origin:
                                continue
                            window = y[:origin]
                            window = window / np.max(window)

                            # 변환
                            rp_img = transform_rp(window)
                            gasf_img = transform_gaf(window, method="summation")
                            gadf_img = transform_gaf(window, method="difference")
                            gaf_2ch = np.stack([gasf_img, gadf_img], axis=-1)
                            rp_stream = rp_img
                            gaf_stream = gaf_2ch

                            # 저장
                            save_base = os.path.join(
                                SAVE_DIR,
                                f"forecast_{cycle}",
                                f"fold_{fold}",
                                f"origin_{origin}"
                            )

                            # RP
                            rp_dir = os.path.join(save_base, "RP", split)
                            os.makedirs(rp_dir, exist_ok=True)
                            np.save(os.path.join(rp_dir, file.replace(".csv", ".npy")), rp_img)

                            # GASF
                            gasf_dir = os.path.join(save_base, "GASF", split)
                            os.makedirs(gasf_dir, exist_ok=True)
                            np.save(os.path.join(gasf_dir, file.replace(".csv", ".npy")), gasf_img)

                            # GADF
                            gadf_dir = os.path.join(save_base, "GADF", split)
                            os.makedirs(gadf_dir, exist_ok=True)
                            np.save(os.path.join(gadf_dir, file.replace(".csv", ".npy")), gadf_img)

                            # 2채널
                            gaf2ch_dir = os.path.join(save_base, "GASF_GADF_2ch", split)
                            os.makedirs(gaf2ch_dir, exist_ok=True)
                            np.save(os.path.join(gaf2ch_dir, file.replace(".csv", ".npy")), gaf_2ch)

                            # 2stream
                            gafstream_dir = os.path.join(save_base, "RP_GAF_2stream", split)
                            os.makedirs(gafstream_dir, exist_ok=True)
                            np.savez(
                                os.path.join(gafstream_dir, file.replace(".csv", ".npz")),
                                rp=rp_stream,
                                gaf=gaf_stream
                            )

                            file_count += 1

                            # 개별 파일 끝나면 메모리 해제
                            del rp_img, gasf_img, gadf_img, gaf_2ch, rp_stream, gaf_stream, df, y, window
                            gc.collect()

                        except Exception as e:
                            print(f"[ERROR] {file} 처리 중 오류 → {e}")
                            traceback.print_exc()
                            continue

                    print(f"[INFO] origin {origin}, split {split} → {file_count}개 저장 완료")

                except Exception as e:
                    print(f"[ERROR] split {split} 전체 처리 실패 → {e}")
                    traceback.print_exc()
                    continue

        # fold 하나 끝날 때 메모리 청소
        gc.collect()
        print(f"\n========== forecast_{cycle} fold_{fold} 완료, 메모리 정리 ==========")

print("\n✅ 전체 forecast_550 변환 작업 끝났습니다!")