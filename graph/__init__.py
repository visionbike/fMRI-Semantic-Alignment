from typing import Any
import torch.nn as nn
import torch.optim as optim


__all__ = [
    "get_optimizer"
    "get_loss",
    "get_network",

]

def get_optimizer(name: str, model: Any, lr: float = 0.001) -> optim.Optimizer:
    if name == "adam":
        optim.Adam(model.parameters(), lr=lr)



def get_loss(name: str) -> nn.Module:
    loss_ = None
    if name == "contrastive":
        from .loss import ContrastiveLoss
        loss_ = ContrastiveLoss()
    elif name == "ce":
        loss_ = nn.CrossEntropyLoss()
    else:
        print("The loss is not supported!")
    return loss_


def get_network(name: str, num_classes: int) -> Any:
    model_ = None
    if name == "cliptuner":
        from .network import CLIPTuner
        model_ = CLIPTuner(num_classes)
    elif name == "pretrain":
        import mobileclip
        model_, _, _ = mobileclip.create_model_and_transforms("MobileCLIP-B", "datacompdr_lt")
    else:
        print("The network is not supported!")
    return model_

