from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as fn

__all__ = [
    "ContrastiveLoss",
    "CombineLoss"
]


class ContrastiveLoss(nn.Module):
    def __init__(self, m: float = 2.0) -> None:
        super().__init__()
        self.m = m  # margin or radius

    def forward(
            self,
            y11: torch.Tensor,
            y12: torch.Tensor,
            y21: torch.Tensor,
            y22: torch.Tensor,
            contrastive: torch.Tensor
    ) -> torch.Tensor:
        euc_dist1 = fn.pairwise_distance(y11, y12, keepdim=True)
        euc_dist2 = fn.pairwise_distance(y21, y22, keepdim=True)

        out1 = torch.mean((1 - contrastive) * torch.pow(euc_dist1, 2) + contrastive * torch.pow(
            torch.clamp(self.m - euc_dist1, min=0., max=None), 2))
        out2 = torch.mean((1 - contrastive) * torch.pow(euc_dist2, 2) + contrastive * torch.pow(
            torch.clamp(self.m - euc_dist2, min=0., max=None), 2))
        out = (out1 + out2) / 2.
        return out


class CombineLoss(nn.Module):
    def __init__(self, co_ratio: float = 0.01, ce_ratio: float = 1.) -> None:
        super().__init__()
        self.co = ContrastiveLoss()
        self.ce = nn.CrossEntropyLoss()
        self.co_ratio = co_ratio
        self.ce_ratio = ce_ratio

    def forward(
            self,
            y11: torch.Tensor,
            y12: torch.Tensor,
            y21: torch.Tensor,
            y22: torch.Tensor,
            out: torch.Tensor,
            tgt: torch.Tensor,
            contrastive: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        # contrastive loss
        out_co = self.co(y11, y12, y21, y22, contrastive)
        # ce loss
        out_ce = self.ce(out, tgt)
        out = self.co_ratio * out_co + self.ce_ratio * out_ce
        return out, out_co, out_ce
