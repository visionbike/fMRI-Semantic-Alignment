from typing import Tuple
from argparse import Namespace
import torch
import torch.utils.tran
import torchvision.transforms.v2 as vtf2
from torch.utils.data import DataLoader

__all__ = ["get_dataloaders"]

def get_dataloaders(args: Namespace) -> Tuple[DataLoader, ...]:
    transform_image = vtf2.Compose([
        vtf2.ToImage(),
        vtf2.ToDtype(torch.float32, scale=True),
        vtf2.Resize((224, 224), interpolation=vtf2.InterpolationMode.BILINEAR),
        vtf2.CenterCrop((224, 224)),
        vtf2.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    loader_train, loader_val, loader_test = None, None, None
    if args.dataset == "BOLD5000":
        from .bold5000 import BOLD5000
        dataset_train = BOLD5000(path_data=args.data_path, mode="train", tf_image=transform_image)
        dataset_val = BOLD5000(path_data=args.data_path, mode="val", tf_image=transform_image)
        dataset_test = BOLD5000(path_data=args.data_path, mode="test", tf_image=transform_image)
        loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        loader_val = DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True)
        loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    else:
        print("The dataset is not supported!")
    return loader_train, loader_val, loader_test
