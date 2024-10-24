import torch.nn as nn
import torch.nn.functional as fn

__all__ = [
    "ContrastiveLoss"
]


class ContrastiveLoss(nn.Module):
    def __init__(self, m: float = 2.0) -> None:
        super().__init__()
        self.m = m # margin or radius

    def forward(
            self,
            y11: torch.Tensor,
            y12: torch.Tensor,
            y21: torch.Tensor,
            y22: torch.Tensor,
            contrastive: int = 0
    ) -> torch.Tensor:
        euc_dist1 = fn.pairwise_distance(y11, y12)
        euc_dist2 = fn.pairwise_distance(y21, y22)
        if contrastive == 0:
            # positive pairs
            out = (torch.mean(torch.pow(euc_dist1) + torch.mean(torch.pow(euc_dist2)))) / 2.
        else:
            # negative pairs
            delta1 = self.m - euc_dist1
            delta2 = self.m - euc_dist2
            delta1 = torch.clamp(delta1, min=0., max=None)
            delta2 = torch.clamp(delta2, min=0., max=None)
            out = (torch.mean(torch.pow(delta1) + torch.mean(torch.pow(delta2)))) / 2.
        return out
