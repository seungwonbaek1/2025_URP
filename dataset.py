import torch
from torch.utils.data import Dataset
import numpy as np
import os

class BatterySOHDataset(Dataset):
    def __init__(self, file_list, label_dict, mode="1ch"):
        self.file_list = file_list
        self.label_dict = label_dict
        self.mode = mode

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        if self.mode in ["1ch", "2ch"]:
            data = np.load(file_path)
            data = torch.tensor(data, dtype=torch.float32)
            if self.mode == "1ch":
                data = data.unsqueeze(0)  # (1, H, W)
            else:
                data = data.permute(2, 0, 1)  # (2, H, W)
        elif self.mode == "2stream":
            npzfile = np.load(file_path)
            rp = npzfile['rp']
            gaf = npzfile['gaf']
            rp = torch.tensor(rp, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
            gaf = torch.tensor(gaf, dtype=torch.float32).permute(2, 0, 1)  # (2, H, W)
            data = (rp, gaf)
        else:
            raise ValueError("mode should be '1ch', '2ch', or '2stream'")

        label = self.label_dict[file_name]
        label = torch.tensor(label, dtype=torch.float32)

        return data, label
