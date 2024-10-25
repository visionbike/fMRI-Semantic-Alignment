from typing import Any
import torch.nn as nn
import torch.optim as optim

__all__ = [
    "get_optimizer",
    "get_loss",
    "get_network",

]

def get_optimizer(name: str, scheduler: str, parameters: Any, lr: float = 0.001) -> optim.Optimizer:
    optimizer = None
    if name == "adam":
        optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    lr_scheduler = None
    if scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        print("The LR scheduler is not supported now !!!")
    return optimizer, lr_scheduler

def get_loss(name: str) -> nn.Module:
    loss_ = None
    if name == "contrastive":
        from .loss import ContrastiveLoss
        loss_ = ContrastiveLoss()
    elif name == "ce":
        loss_ = nn.CrossEntropyLoss()
    elif name == "combine":
        from .loss import CombineLoss
        loss_ = CombineLoss()
    else:
        print("The loss is not supported!")
    return loss_


def get_network(name: str, num_classes: int) -> Any:
    model_ = None
    tokenizer_ = None
    if name == "mri":
        from .network import MRIModel
        model_ = MRIModel(num_classes)
        tokenizer_ = None
    elif name == "clip":
        import open_clip
        model_ = open_clip.create_model("MobileCLIP-B", "datacompdr_lt")
        tokenizer_ = open_clip.get_tokenizer("MobileCLIP-B")
    else:
        print("The network is not supported!")
    return model_, tokenizer_
