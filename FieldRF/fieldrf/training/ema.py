"""Floating-point EMA for model evaluation and checkpoint selection."""

import torch


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    @torch.no_grad()
    def apply_to(self, model):
        """Swap EMA weights into model. Returns backup dict for restore()."""
        backup = {}
        sd = model.state_dict()
        for k, v in self.shadow.items():
            if k in sd and sd[k].shape == v.shape:
                backup[k] = sd[k].clone()
                sd[k].copy_(v)
        return backup

    @torch.no_grad()
    def restore(self, model, backup):
        """Restore original weights from backup dict."""
        sd = model.state_dict()
        for k, v in backup.items():
            if k in sd:
                sd[k].copy_(v)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        for k in list(self.shadow.keys()):
            if k in sd and sd[k].shape == self.shadow[k].shape:
                self.shadow[k].copy_(sd[k])
