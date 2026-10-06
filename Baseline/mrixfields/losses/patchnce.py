"""PatchNCE contrastive loss for CUT (Contrastive Unpaired Translation).

Based on the official CUT implementation:
    taesungp/contrastive-unpaired-translation/models/patchnce.py

Reference: Park et al., "Contrastive Learning for Unpaired Image-to-Image
Translation", ECCV 2020.

Includes:
    - Standard PatchNCE loss
    - MoCo MemoryBank for anti-collapse (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryBank(nn.Module):
    """Circular queue of projected features for MoCo-style contrastive learning.

    Stores L2-normalized features from past iterations as additional negatives
    for the PatchNCE loss. Prevents generator output collapse by maintaining
    diverse cross-iteration negative samples.

    Args:
        queue_size: Maximum number of feature vectors to store.
        feat_dim: Dimension of projected features (netF_nc).
        n_layers: Number of NCE layers (one queue per layer).
    """

    def __init__(self, queue_size: int, feat_dim: int, n_layers: int):
        super().__init__()
        self.queue_size = queue_size
        self.feat_dim = feat_dim
        self.n_layers = n_layers
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("full_flag", torch.zeros(1, dtype=torch.bool))

        # One queue per NCE layer: (queue_size, feat_dim)
        for i in range(n_layers):
            self.register_buffer(
                f"queue_{i}",
                F.normalize(torch.randn(queue_size, feat_dim), dim=1),
            )

    def _get_queue(self, i: int) -> torch.Tensor:
        return getattr(self, f"queue_{i}")

    def enqueue(self, feats_per_layer: list):
        """Push new features into the queue (FIFO).

        Args:
            feats_per_layer: list of [num_patches, feat_dim] tensors,
                one per NCE layer, already L2-normalized.
        """
        n = feats_per_layer[0].size(0)
        ptr = self.ptr.item()

        if ptr + n <= self.queue_size:
            for i, f in enumerate(feats_per_layer):
                self._get_queue(i)[ptr:ptr + n] = f.detach()
        else:
            remaining = self.queue_size - ptr
            for i, f in enumerate(feats_per_layer):
                self._get_queue(i)[ptr:] = f.detach()[:remaining]
                self._get_queue(i)[:n - remaining] = f.detach()[remaining:]

        new_ptr = (ptr + n) % self.queue_size
        self.ptr[0] = new_ptr
        if new_ptr == 0:
            self.full_flag[0] = True

    def get_all(self) -> list:
        """Return all queues (list of [queue_size, feat_dim] tensors)."""
        return [self._get_queue(i) for i in range(self.n_layers)]

    def is_ready(self) -> bool:
        return bool(self.full_flag[0].item())


class PatchNCELoss(nn.Module):
    """Patch-wise InfoNCE loss for contrastive unpaired translation.

    Given query features (from generated image) and key features (from real
    source image) at corresponding spatial locations, treats same-position
    pairs as positives and all other positions as negatives.

    When ``bank`` is provided (list of [queue_size, dim] per layer from a
    MemoryBank), all bank vectors are also used as negatives, providing
    diverse cross-iteration negative samples (MoCo style).

    Parameters:
        nce_T (float) -- temperature for softmax (default 0.07)
        batch_size (int) -- batch size (used when not including all negatives)
        nce_includes_all_negatives_from_minibatch (bool) -- if True, negatives
            from other samples in the batch are included
    """

    def __init__(self, nce_T=0.07, batch_size=1,
                 nce_includes_all_negatives_from_minibatch=False):
        super().__init__()
        self.nce_T = nce_T
        self.batch_size = batch_size
        self.nce_includes_all_negatives_from_minibatch = nce_includes_all_negatives_from_minibatch
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k, bank=None):
        """Compute PatchNCE loss.

        Args:
            feat_q: Query features from generated image, shape (num_patches, dim).
            feat_k: Key features from source image, shape (num_patches, dim).
            bank: Optional tensor of bank negatives, shape (queue_size, dim).

        Returns:
            Per-patch NCE loss, shape (num_patches,).
        """
        num_patches = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # Positive logit: dot product of corresponding positions
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1),
            feat_k.view(num_patches, -1, 1),
        )
        l_pos = l_pos.view(num_patches, 1)

        # Negative logits from current batch
        if self.nce_includes_all_negatives_from_minibatch:
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.batch_size

        feat_q_reshaped = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k_reshaped = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q_reshaped.size(1)
        l_neg_curbatch = torch.bmm(feat_q_reshaped, feat_k_reshaped.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        # MoCo: additional negatives from the memory bank
        if bank is not None:
            l_neg_bank = torch.mm(feat_q, bank.T)  # (num_patches, queue_size)
            l_neg = torch.cat([l_neg, l_neg_bank], dim=1)

        out = torch.cat((l_pos, l_neg), dim=1) / self.nce_T

        loss = self.cross_entropy_loss(
            out, torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
        )

        return loss

__all__ = [
    "PatchNCELoss",
    "MemoryBank",
]
