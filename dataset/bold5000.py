from typing import Optional, Callable, Dict
from pathlib import Path
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class BOLD5000(Dataset):
    def __init__(
            self,
            path_data: str,
            mode: str = "train",
            tf_image: Optional[Callable] = None
    ) -> None:
        self.path_data = Path(path_data)
        self.map_label = np.load(self.path_data / "label_map.npy").tolist()
        self.map_stimuli = np.load(self.path_data / "stimuli_map.npy", allow_pickle=True).tolist()
        self.data = np.load(self.path_data / f"{mode}.npy", allow_pickle=True).tolist()
        self.tf_image = tf_image

    def __len__(self) -> int:
        return len(self.data["label"])

    def __getitem__(self, idx: int) -> Dict:
        mri = self.data["data"][idx]
        tgt = self.data["label"][idx]
        if torch.rand(1) >= 0.5:
            # get positive image
            fn_img = random.choice(self.map_stimuli[tgt])
            cls_name = self.map_label[tgt]
            contrastive = 0
        else:
            map_label_neg = self.map_label.copy()
            map_label_neg.remove(self.map_label[tgt])
            cls_name = random.choice(map_label_neg)
            tgt_neg = self.map_label.index(cls_name)
            fn_img = random.choice(self.map_stimuli[tgt_neg])
            contrastive = 1
        img = cv2.imread(fn_img, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.tf_image:
            img = self.tf_image(img)
        mri = torch.from_numpy(mri).float()
        tgt = torch.tensor(tgt).long()
        contrastive = torch.tensor(contrastive).float()
        return {"mri": mri, "img": img, "tgt": tgt, "cls_name": cls_name, "contrastive": contrastive}
