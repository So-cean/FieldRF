"""UNSB model: Neural Schrödinger Bridge for unpaired image translation.

Adapted from:
    https://github.com/cyclomon/UNSB
    Kim et al., ICLR 2024.

Loss = lambda_GAN * GAN(G(Xt)) + lambda_NCE * PatchNCE(G(Xt), X)
       + lambda_SB * SchrodingerBridge(G)

The generator is conditioned on (image, timestep, latent_z).
During training, a forward SDE interpolates from source to target domain
via multi-step refinement.
"""

import numpy as np
import torch
import torch.nn as nn

from .net_init import init_net
from .networks_unsb import PatchSampleF, ResnetGenerator_ncsn, NLayerDiscriminator_ncsn
from ..losses.adversarial import GANLoss
from ..losses.patchnce import PatchNCELoss
from ..losses.tokennce import TokenNCELoss


class UNSBModel(nn.Module):
    """UNSB: Unpaired Neural Schrodinger Bridge.

    Parameters:
        input_nc (int)      -- input channels
        output_nc (int)     -- output channels
        ndf (int)           -- discriminator filters
        ngf (int)           -- generator filters
        n_blocks (int)      -- ResNet blocks in generator
        n_mlp (int)         -- mapping network layers for z
        z_dim (int)         -- latent dimension
        t_emb_dim (int)     -- time embedding dimension
        num_timesteps (int) -- number of Schrodinger bridge steps (T)
        tau (float)         -- entropy regularization
        std (float)         -- Gaussian noise scale for D
        lambda_GAN (float)-- GAN loss weight
        lambda_NCE (float)-- NCE loss weight
        lambda_SB (float) -- Schrodinger Bridge loss weight
        nce_layers (list) -- layer indices for NCE
        nce_T (float)     -- NCE temperature
        nce_idt (bool)    -- use identity NCE
        gan_mode (str)    -- 'lsgan' or 'vanilla'
        lr (float)        -- learning rate
        beta1, beta2 (float)-- Adam betas
        patch_mode (str)    -- 'patchnce' or 'tokennce', NCE loss type
    """

    def __init__(
        self,
        input_nc=1,
        output_nc=1,
        ndf=64,
        ngf=64,
        n_blocks=9,
        n_mlp=3,
        z_dim=256,
        t_emb_dim=256,
        num_timesteps=5,
        tau=0.01,
        std=0.25,
        lambda_GAN=1.0,
        lambda_NCE=1.0,
        lambda_idt=None,
        lambda_SB=1.0,
        nce_layers=None,
        nce_T=0.07,
        nce_idt=True,
        gan_mode="lsgan",
        lr=0.0002,
        beta1=0.5,
        beta2=0.999,
        patch_mode="patchnce",
        num_patches=256,
        batch_size=1,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.tau = tau
        self.std = std
        self.lambda_GAN = lambda_GAN
        self.lambda_NCE = lambda_NCE
        self.lambda_idt = lambda_NCE if lambda_idt is None else lambda_idt
        self.lambda_SB = lambda_SB
        self.nce_idt = nce_idt
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.patch_mode = patch_mode.lower()
        self.num_patches = num_patches

        if nce_layers is None:
            nce_layers = [0, 4, 8, 12, 16]
        self.nce_layers = nce_layers

        # Precompute Schrodinger Bridge time schedule
        incs = np.array([0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)])
        times = np.cumsum(incs)
        times = times / times[-1]
        times = 0.5 * times[-1] + 0.5 * times
        times = np.concatenate([np.zeros(1), times])
        self.register_buffer("sb_times", torch.tensor(times, dtype=torch.float32))

        # Networks
        # Use InstanceNorm (no running stats) to avoid inplace buffer updates
        # that break DDP backward when D is called twice per iteration.
        norm_layer = nn.InstanceNorm2d

        self.netG = ResnetGenerator_ncsn(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            norm_layer=norm_layer,
            n_blocks=n_blocks,
            n_mlp=n_mlp,
            z_dim=z_dim,
            t_emb_dim=t_emb_dim,
        )
        # Store z_dim on model directly so DDP wrapper doesn't block access
        self.z_dim = z_dim
        self.t_emb_dim = t_emb_dim

        self.netD = NLayerDiscriminator_ncsn(
            input_nc=output_nc,
            ndf=ndf,
            n_layers=3,
            norm_layer=norm_layer,
            t_emb_dim=t_emb_dim,
        )
        self.netE = NLayerDiscriminator_ncsn(
            input_nc=output_nc * 4,  # forward() concats input+input2, each is output_nc*2
            ndf=ndf,
            n_layers=3,
            norm_layer=norm_layer,
            t_emb_dim=t_emb_dim,
        )

        # PatchSampleF for NCE (samples random spatial locations from feature maps)
        self.netF = PatchSampleF(use_mlp=True, nc=256)

        # Losses
        self.criterionGAN = GANLoss(mode=gan_mode)
        if self.patch_mode == "tokennce":
            self.criterionNCE = nn.ModuleList([
                TokenNCELoss(nce_T=nce_T) for _ in self.nce_layers
            ])
        else:
            self.criterionNCE = nn.ModuleList([
                PatchNCELoss(batch_size=batch_size, nce_T=nce_T) for _ in self.nce_layers
            ])

    def init_weights(self, init_type="normal", init_gain=0.02):
        init_net(self.netG, init_type, init_gain)
        init_net(self.netD, init_type, init_gain)
        init_net(self.netE, init_type, init_gain)

    def setup_optimizers(self):
        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        self.optimizer_E = torch.optim.Adam(
            self.netE.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )
        # optimizer_F is created in data_dependent_initialize after netF MLPs are built

    def data_dependent_initialize(self, real_A, real_A2, real_B):
        """Initialize netF MLPs based on actual feature dimensions.

        Must be called once with real data before training starts.
        """
        self.forward(real_A, real_A2, real_B)
        self.compute_D_loss(real_B).backward()
        self.compute_G_loss(real_A, real_B).backward()
        self.optimizer_F = torch.optim.Adam(
            self.netF.parameters(), lr=self.lr, betas=(self.beta1, self.beta2)
        )

    # --------------------------------------------------------------------- #
    #  Forward SDE: generate noisy intermediate Xt
    # --------------------------------------------------------------------- #
    def forward_sde(self, real_A, real_A2=None, real_B=None):
        """Run forward Schrodinger Bridge SDE to generate noisy intermediates.

        Args:
            real_A:  source domain image (B, C, H, W)
            real_A2: second source sample for SB energy (B, C, H, W)
            real_B:  target domain image for identity (B, C, H, W)

        Returns:
            real_A_noisy: Xt at sampled timestep (B, C, H, W)
            real_A_noisy2: Xt2 for SB energy (B, C, H, W)
            XtB: Xt for identity (B, C, H, W) if nce_idt
            time_idx: sampled timestep (B,)  (here always same for whole batch)
        """
        B = real_A.size(0)
        T = self.num_timesteps
        times = self.sb_times
        device = real_A.device

        # Sample a single timestep for the batch (matching original UNSB)
        t_sample = torch.randint(T, size=(1,), device=device).long()
        time_idx = t_sample.expand(B)

        with torch.no_grad():
            self.netG.eval()
            # First trajectory: real_A
            Xt = None
            for t in range(t_sample.item() + 1):
                if t > 0:
                    delta = times[t] - times[t - 1]
                    denom = times[-1] - times[t - 1]
                    inter = (delta / denom).view(1, 1, 1, 1)
                    scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                    noise = torch.randn_like(Xt).to(device)
                    Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * self.tau).sqrt() * noise
                else:
                    Xt = real_A

                t_idx = (t * torch.ones(B, device=device)).long()
                z = torch.randn(B, self.z_dim, device=device)
                Xt_1 = self.netG(Xt, t_idx, z)
            real_A_noisy = Xt.detach()

            # Second trajectory for SB energy
            real_A_noisy2 = None
            if real_A2 is not None and self.lambda_SB > 0:
                Xt2 = None
                for t in range(t_sample.item() + 1):
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).view(1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                        noise = torch.randn_like(Xt2).to(device)
                        Xt2 = (1 - inter) * Xt2 + inter * Xt_12.detach() + (scale * self.tau).sqrt() * noise
                    else:
                        Xt2 = real_A2

                    t_idx = (t * torch.ones(B, device=device)).long()
                    z = torch.randn(B, self.z_dim, device=device)
                    Xt_12 = self.netG(Xt2, t_idx, z)
                real_A_noisy2 = Xt2.detach()

            # Identity trajectory
            XtB = None
            if self.nce_idt and real_B is not None:
                XtB_tmp = None
                for t in range(t_sample.item() + 1):
                    if t > 0:
                        delta = times[t] - times[t - 1]
                        denom = times[-1] - times[t - 1]
                        inter = (delta / denom).view(1, 1, 1, 1)
                        scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                        noise = torch.randn_like(XtB_tmp).to(device)
                        XtB_tmp = (1 - inter) * XtB_tmp + inter * Xt_1B.detach() + (scale * self.tau).sqrt() * noise
                    else:
                        XtB_tmp = real_B

                    t_idx = (t * torch.ones(B, device=device)).long()
                    z = torch.randn(B, self.z_dim, device=device)
                    Xt_1B = self.netG(XtB_tmp, t_idx, z)
                XtB = XtB_tmp.detach()

        return real_A_noisy, real_A_noisy2, XtB, time_idx

    # --------------------------------------------------------------------- #
    #  Main forward (training)
    # --------------------------------------------------------------------- #
    def forward(self, real_A, real_A2=None, real_B=None):
        """Run forward pass.

        Returns:
            fake_B: generated target (B, C, H, W)
            real_A_noisy: noisy source intermediate (B, C, H, W)
            time_idx: timestep indices (B,)
        """
        real_A_noisy, real_A_noisy2, XtB, time_idx = self.forward_sde(real_A, real_A2, real_B)
        self.real_A_noisy = real_A_noisy
        self.real_A_noisy2 = real_A_noisy2
        self.XtB = XtB
        self.time_idx = time_idx

        B = real_A.size(0)
        z = torch.randn(B, self.z_dim, device=real_A.device)

        # Training forward: noisy source -> generator -> prediction
        self.fake_B = self.netG(real_A_noisy, time_idx, z)

        if self.lambda_SB > 0 and real_A2 is not None:
            z2 = torch.randn(B, self.z_dim, device=real_A.device)
            self.fake_B2 = self.netG(real_A_noisy2, time_idx, z2)
        else:
            self.fake_B2 = None

        if self.nce_idt and XtB is not None:
            z_idt = torch.randn(B, self.z_dim, device=real_A.device)
            self.idt_B = self.netG(XtB, time_idx, z_idt)
        else:
            self.idt_B = None

        return self.fake_B

    # --------------------------------------------------------------------- #
    #  Loss computations
    # --------------------------------------------------------------------- #
    def compute_D_loss(self, real_B):
        """GAN loss for discriminator D."""
        fake = self.fake_B.detach()
        pred_fake = self.netD(fake, self.time_idx)
        loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        pred_real = self.netD(real_B, self.time_idx)
        loss_D_real = self.criterionGAN(pred_real, True).mean()
        return (loss_D_fake + loss_D_real) * 0.5

    def compute_E_loss(self):
        """Energy network E loss (for Schrodinger Bridge).

        Original UNSB formulation with temp**2 regularization.
        """
        if self.lambda_SB <= 0.0 or self.fake_B2 is None:
            return torch.tensor(0.0, device=self.fake_B.device)

        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B.detach()], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2.detach()], dim=1)

        pos_energy = self.netE(XtXt_1, self.time_idx, XtXt_1)
        neg_energy = self.netE(XtXt_1, self.time_idx, XtXt_2)

        temp = torch.logsumexp(neg_energy.reshape(-1), dim=0)
        loss_E = -pos_energy.mean() + temp + temp ** 2
        return loss_E

    def compute_G_loss(self, real_A, real_B):
        """GAN + NCE + SB loss for generator."""
        fake = self.fake_B
        loss_G_GAN = 0.0
        if self.lambda_GAN > 0.0:
            pred_fake = self.netD(fake, self.time_idx)
            loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.lambda_GAN

        # Schrodinger Bridge loss
        loss_SB = 0.0
        if self.lambda_SB > 0.0 and self.fake_B2 is not None:
            XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
            XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

            pos = self.netE(XtXt_1, self.time_idx, XtXt_1).mean()
            neg = torch.logsumexp(
                self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0
            )
            ET_XY = pos - neg
            loss_SB = -((self.num_timesteps - self.time_idx[0]) / self.num_timesteps *
                        self.tau * ET_XY)
            loss_SB += self.tau * torch.mean((self.real_A_noisy - self.fake_B) ** 2)

        # NCE loss
        loss_NCE = 0.0
        if self.lambda_NCE > 0.0:
            loss_NCE = self.calculate_NCE_loss(real_A, fake)

        loss_NCE_both = loss_NCE
        if self.nce_idt and self.lambda_NCE > 0.0 and self.idt_B is not None:
            loss_NCE_Y = self.calculate_NCE_loss(real_B, self.idt_B)
            idt_scale = self.lambda_idt / self.lambda_NCE
            loss_NCE_both = (loss_NCE + loss_NCE_Y * idt_scale) * 0.5

        return loss_G_GAN + self.lambda_SB * loss_SB + self.lambda_NCE * loss_NCE_both

    def calculate_NCE_loss(self, src, tgt, nce_weight=None):
        """PatchNCE / TokenNCE between src and tgt.

        Uses PatchSampleF to randomly sample num_patches spatial locations
        from encoder feature maps, matching the original CUT/UNSB design.
        """
        B = src.size(0)
        z = torch.randn(B, self.z_dim, device=src.device)
        zero_t = torch.zeros(B, device=src.device, dtype=torch.long)

        feat_q = self.netG(tgt, zero_t, z, layers=self.nce_layers, encode_only=True)
        feat_k = self.netG(src, zero_t, z, layers=self.nce_layers, encode_only=True)

        # Use PatchSampleF to randomly sample patches (avoids OOM from full spatial)
        feat_k_pool, sample_ids = self.netF(feat_k, self.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.num_patches, sample_ids)

        total = 0.0
        n_layers = len(self.nce_layers)
        for f_q, f_k, crit in zip(feat_q_pool, feat_k_pool, self.criterionNCE):
            loss = crit(f_q, f_k)
            total += loss.mean()
        return total / n_layers

    # --------------------------------------------------------------------- #
    #  Full optimization step
    # --------------------------------------------------------------------- #
    def optimize_parameters(self, real_A, real_A2, real_B, max_grad_norm=None):
        """Full training step.

        Args:
            real_A: source batch
            real_A2: second source batch (for SB energy)
            real_B: target batch
            max_grad_norm: gradient clipping

        Returns:
            Dict of loss values.
        """
        # Forward
        self.forward(real_A, real_A2, real_B)

        # Update D
        for p in self.netD.parameters():
            p.requires_grad = True
        self.optimizer_D.zero_grad()
        loss_D = self.compute_D_loss(real_B)
        loss_D.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_grad_norm)
        self.optimizer_D.step()

        # Update E
        if self.lambda_SB > 0.0:
            for p in self.netE.parameters():
                p.requires_grad = True
            self.optimizer_E.zero_grad()
            loss_E = self.compute_E_loss()
            loss_E.backward()
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.netE.parameters(), max_grad_norm)
            self.optimizer_E.step()
        else:
            loss_E = torch.tensor(0.0, device=real_A.device)

        # Update G (+ F if using MLP projection)
        netF = self.netF.module if hasattr(self.netF, "module") else self.netF
        for p in self.netD.parameters():
            p.requires_grad = False
        for p in self.netE.parameters():
            p.requires_grad = False
        self.optimizer_G.zero_grad()
        if netF.use_mlp:
            self.optimizer_F.zero_grad()
        loss_G = self.compute_G_loss(real_A, real_B)
        loss_G.backward()
        if max_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_grad_norm)
            if netF.use_mlp:
                torch.nn.utils.clip_grad_norm_(self.netF.parameters(), max_grad_norm)
        self.optimizer_G.step()
        if netF.use_mlp:
            self.optimizer_F.step()

        return {
            "loss_D": loss_D.item(),
            "loss_E": loss_E.item(),
            "loss_G": loss_G.item(),
        }

    # --------------------------------------------------------------------- #
    #  Inference (multi-step refinement)
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def inference(self, real_A, num_steps=None):
        """Run multi-step Schrodinger Bridge inference.

        Args:
            real_A: source image (B, C, H, W)
            num_steps: number of refinement steps (default: self.num_timesteps)

        Returns:
            List of intermediate outputs [Xt_1_step1, Xt_1_step2, ...]
        """
        self.netG.eval()
        T = num_steps or self.num_timesteps
        times = self.sb_times
        B = real_A.size(0)
        device = real_A.device

        outputs = []
        Xt = real_A
        for t in range(T):
            if t > 0:
                delta = times[t] - times[t - 1]
                denom = times[-1] - times[t - 1]
                inter = (delta / denom).view(1, 1, 1, 1)
                scale = (delta * (1 - delta / denom)).view(1, 1, 1, 1)
                noise = torch.randn_like(Xt).to(device)
                Xt = (1 - inter) * Xt + inter * Xt_1.detach() + (scale * self.tau).sqrt() * noise

            t_idx = (t * torch.ones(B, device=device)).long()
            z = torch.randn(B, self.z_dim, device=device)
            Xt_1 = self.netG(Xt, t_idx, z)
            outputs.append(Xt_1)

        return outputs

__all__ = [
    "UNSBModel",
]
