import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 논문 구조 CNN (RP, GASF, GADF, 2ch용 공통 구조)
# - CAM 시각화 가능하도록 fc 포함
# - TwoStreamCNN과 호환되도록 return_feature 옵션 추가
# ------------------------------
class CNN_Professor(nn.Module):
    def __init__(self, in_channels=1, return_feature=False):
        super(CNN_Professor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=11)
        self.pool1 = nn.AvgPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=3)
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 32, 1, 1)
        self.fc = nn.Linear(32, 1)          # CAM 생성을 위한 FC layer
        self.return_feature = return_feature

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # (B, 32)

        if self.return_feature:
            return x  # feature vector만 반환 (TwoStreamCNN용)
        else:
            return self.fc(x)  # 최종 예측값 반환 (CAM 생성용)


# ------------------------------
# 논문 구조 Two-stream CNN (RP + GAF)
# - stream_rp, stream_gaf에서 feature만 추출
# - 최종 fc_final로 예측값 생성
# ------------------------------
class TwoStreamCNN_Professor(nn.Module):
    def __init__(self):
        super(TwoStreamCNN_Professor, self).__init__()
        self.stream_rp = CNN_Professor(in_channels=1, return_feature=True)
        self.stream_gaf = CNN_Professor(in_channels=2, return_feature=True)
        self.fc_final = nn.Linear(64, 1)  # 32 + 32 → 1

    def forward(self, x):
        x_rp, x_gaf = x
        f_rp = self.stream_rp(x_rp)     # (B, 32)
        f_gaf = self.stream_gaf(x_gaf)  # (B, 32)
        fused = torch.cat([f_rp, f_gaf], dim=1)  # (B, 64)
        out = self.fc_final(fused)     # (B, 1)
        return out
