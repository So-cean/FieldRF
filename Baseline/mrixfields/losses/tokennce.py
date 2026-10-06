"""TokenNCE contrastive loss for Transformer-based unpaired image translation.

Compatible interface with PatchNCELoss (Park et al., ECCV 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenNCELoss(nn.Module):
    """Token-wise InfoNCE loss for sequence generators.

    Replaces PatchNCELoss in CUT models. Supports optional relation contrast
    for explicit structural topology supervision (impossible in CNN).

    Parameters:
        nce_T (float) -- temperature for softmax (default 0.07)
        batch_size (int) -- reserved for interface compatibility
        nce_includes_all_negatives_from_minibatch (bool) -- reserved for interface compatibility
        lambda_rel (float) -- weight for relation branch (default 1.0)
    """

    def __init__(self, nce_T=0.07, batch_size=1,
                 nce_includes_all_negatives_from_minibatch=False,
                 lambda_rel=1.0):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.lambda_rel = lambda_rel
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k, feat_r_q=None, feat_r_k=None):
        """Compute TokenNCE loss.

        Args:
            feat_q: Query features, shape (B, N, C) or (num_patches, C).
            feat_k: Key features. Will be detached. Same shape as feat_q.
            feat_r_q: Optional relation query, shape (B, N, K) or (num_patches, K).
            feat_r_k: Optional relation key. Will be detached. Same shape as feat_r_q.

        Returns:
            Per-token NCE loss, shape (B*N,) or (num_patches,).
        """
        # 1. Semantic identity contrast
        loss = self._info_nce(feat_q, feat_k)

        # 2. Structural topology contrast (IMPOSSIBLE in CNN)
        if feat_r_q is not None and feat_r_k is not None:
            loss_rel = self._info_nce(feat_r_q, feat_r_k)
            loss = loss + self.lambda_rel * loss_rel

        return loss

    def _info_nce(self, q, k):
        """Strict diagonal InfoNCE. Memory-optimized vs naive sim-diagonal-cat.

        q, k: (B, N, C) or (num_patches, C)
        """
        single_batch = False
        if q.dim() == 2:
            q = q.unsqueeze(0)
            k = k.unsqueeze(0)
            single_batch = True

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1).detach()
        B, N, D = q.shape

        # 正样本: (B, N, 1)
        # 用 element-wise mul + sum 替代 bmm-diagonal，避免 materialize (B,N,N)
        l_pos = (q * k).sum(dim=-1, keepdim=True) / self.nce_T

        # 负样本: 复用 bmm 输出
        l_neg = torch.bmm(q, k.transpose(1, 2)) / self.nce_T
        mask = torch.eye(N, device=q.device, dtype=self.mask_dtype).unsqueeze(0)
        l_neg.masked_fill_(mask, -10.0)
        
        # 合并: (B, N, 1+N)
        out = torch.cat((l_pos, l_neg), dim=-1)

        # CrossEntropy: target=0 means first column (l_pos) is positive
        target = torch.zeros(B * N, dtype=torch.long, device=q.device)
        loss = self.cross_entropy_loss(out.view(-1, N + 1), target)

        if single_batch:
            return loss.view(N)
        return loss.view(B, N)
__all__ = [
    "TokenNCELoss",
]
