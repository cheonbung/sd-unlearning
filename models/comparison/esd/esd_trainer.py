"""Canonical ESD trainer (Erased Stable Diffusion; Gandikota et al., ICCV 2023).

Faithful to the official recipe (github.com/rohitgandikota/erasing):
  * Frozen original U-Net provides the negative-guidance teacher; only the trainee U-Net subset
    (see esd_params.set_trainable_esd) is updated.
  * Each step: partially denoise from noise with the *trainee* model conditioned on the concept
    prompt to a random DDIM step t (this is the canonical choice -- the model erases its own
    samples), then regress the trainee's concept-conditioned prediction onto the negative-guided
    target computed from the FROZEN model:

        target = e_0 - eta * (e_p - e_0)          # Eq. 6, eta = negative_guidance
        L      = MSE(e_n, target.detach())

    with e_0 = frozen(z,t,uncond), e_p = frozen(z,t,concept), e_n = trainee(z,t,concept).
  * Defaults: ESD-u (noxattn), eta=1, lr=1e-5, 1000 steps, batch 1, Adam. SD v1.4.

NOT an FCFTrainer/ODACETrainer subclass (CLAUDE.md). Saves a standalone U-Net that the xeval
harness swaps in via `kind="esd"` (identical UNET-swap path to ODACE).

VRAM note (12 GB target): ESD-u trains ~85% of the 859M-param U-Net, so fp32 Adam moments alone
would exceed the card. We keep fp32 MASTER weights (numerically faithful) but use bitsandbytes
8-bit (paged) Adam for the optimizer state. Trainee forward/backward in fp32 + gradient
checkpointing; frozen teacher in fp16; trajectory sampling under fp16 autocast.

Callers: models/comparison/esd/train_esd.py, models/comparison/esd/tests/test_esd.py
"""
from __future__ import annotations

import logging
import os
import random
from typing import Dict, List

import torch
import torch.nn.functional as F

from esd_params import set_trainable_esd

try:                                  # progress bar with ETA (optional dep)
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

logger = logging.getLogger(__name__)


def _build_optimizer(params, lr: float):
    """Prefer bitsandbytes paged 8-bit Adam (fits large trainable sets in 12 GB); fall back to
    plain Adam8bit, then torch Adam. Returns (optimizer, name)."""
    try:
        import bitsandbytes as bnb
        try:
            return bnb.optim.PagedAdam8bit(params, lr=lr), "PagedAdam8bit"
        except Exception:  # noqa: BLE001
            return bnb.optim.Adam8bit(params, lr=lr), "Adam8bit"
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"bitsandbytes unavailable ({exc}); using torch.optim.Adam (more VRAM)")
        from torch.optim import Adam
        return Adam(params, lr=lr), "Adam"


class ESDTrainer:
    def __init__(self, sd_model_id: str, device, train_method: str = "noxattn",
                 eta: float = 1.0, learning_rate: float = 1e-5, ddim_steps: int = 50,
                 start_guidance: float = 3.0, negative_guidance: float = 1.0,
                 max_length: int = 77, seed: int = 42, **_ignore):
        from diffusers import UNet2DConditionModel, DDIMScheduler
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device
        self.eta = float(negative_guidance if negative_guidance is not None else eta)
        self.ddim_steps = ddim_steps
        self.start_guidance = start_guidance
        self.max_length = max_length
        random.seed(seed); torch.manual_seed(seed)

        self.tokenizer = CLIPTokenizer.from_pretrained(sd_model_id, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(
            sd_model_id, subfolder="text_encoder").to(device)
        self.text_encoder.requires_grad_(False); self.text_encoder.eval()

        # Trainee U-Net: fp32 master weights for numerically faithful small updates.
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet").to(device)
        try:
            self.unet.enable_gradient_checkpointing()
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"grad checkpointing unavailable: {exc}")
        n_train = set_trainable_esd(self.unet, train_method)
        self.unet.train()

        # Frozen teacher in fp16 (provides e_0, e_p for the negative-guidance target).
        self.unet_frozen = UNet2DConditionModel.from_pretrained(
            sd_model_id, subfolder="unet").to(device=device, dtype=torch.float16)
        self.unet_frozen.requires_grad_(False); self.unet_frozen.eval()

        self.ddim = DDIMScheduler.from_pretrained(sd_model_id, subfolder="scheduler")
        trainable = [p for p in self.unet.parameters() if p.requires_grad]
        self.optimizer, opt_name = _build_optimizer(trainable, learning_rate)
        self.lat_ch = self.unet.config.in_channels
        self._uncond = None
        logger.info(f"[ESD] method={train_method} trainable={n_train:,} params | eta={self.eta} "
                    f"start_g={start_guidance} lr={learning_rate} opt={opt_name}")

    @torch.no_grad()
    def _encode(self, texts: List[str]) -> torch.Tensor:
        tok = self.tokenizer(texts, padding="max_length", max_length=self.max_length,
                             truncation=True, return_tensors="pt").to(self.device)
        return self.text_encoder(tok.input_ids)[0]

    @torch.no_grad()
    def _uncond_emb(self) -> torch.Tensor:
        if self._uncond is None:
            self._uncond = self._encode([""])
        return self._uncond

    @torch.no_grad()
    def _sample_until(self, c_concept: torch.Tensor, t_enc_idx: int):
        """Canonical ESD trajectory: DDIM-denoise from noise with the TRAINEE U-Net + concept
        prompt (CFG, start_guidance) for t_enc_idx steps. Trainee weights are autocast to fp16
        for speed/memory; no grad. Returns (z, t_current)."""
        self.ddim.set_timesteps(self.ddim_steps, device=self.device)
        z = torch.randn(1, self.lat_ch, 64, 64, device=self.device, dtype=torch.float16)
        z = z * self.ddim.init_noise_sigma
        cond = torch.cat([self._uncond_emb(), c_concept]).half()
        tlist = self.ddim.timesteps
        t_cur = tlist[0]
        with torch.autocast("cuda", dtype=torch.float16):
            for i, t in enumerate(tlist):
                if i >= t_enc_idx:
                    break
                t_cur = t
                zin = self.ddim.scale_model_input(torch.cat([z] * 2), t)
                npred = self.unet(zin, t, encoder_hidden_states=cond).sample
                nu, nc = npred.chunk(2)
                npred = nu + self.start_guidance * (nc - nu)
                z = self.ddim.step(npred, t, z).prev_sample
        return z, t_cur

    def _train_step(self, concept: str) -> Dict[str, float]:
        c_concept = self._encode([concept])
        c_uncond = self._uncond_emb()
        t_enc = random.randint(1, self.ddim_steps - 1)
        z, t_cur = self._sample_until(c_concept, t_enc)   # fp16 latent on concept trajectory

        with torch.no_grad():
            e_0 = self.unet_frozen(z, t_cur, encoder_hidden_states=c_uncond.half()).sample.float()
            e_p = self.unet_frozen(z, t_cur, encoder_hidden_states=c_concept.half()).sample.float()
        target = (e_0 - self.eta * (e_p - e_0)).detach()  # Eq. 6 negative guidance

        zf = z.float()
        self.optimizer.zero_grad(set_to_none=True)
        e_n = self.unet(zf, t_cur, encoder_hidden_states=c_concept).sample
        loss = F.mse_loss(e_n, target)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def train(self, concepts: List[str], num_steps: int = 1000, log_every: int = 25) -> List[Dict]:
        if not concepts:
            raise ValueError("ESD needs at least one concept prompt to erase")
        history = []
        bar = tqdm(range(1, num_steps + 1), desc="ESD train", unit="step") if _HAS_TQDM \
            else range(1, num_steps + 1)
        for step in bar:
            concept = random.choice(concepts)
            s = self._train_step(concept); s["step"] = step
            history.append(s)
            if _HAS_TQDM:
                bar.set_postfix(loss=f"{s['loss']:.5f}")
            if step % log_every == 0 or step == 1:
                logger.info(f"  step {step}/{num_steps} | loss={s['loss']:.5f} | concept={concept!r}")
        return history

    def save(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        self.unet.save_pretrained(save_dir)
        logger.info(f"ESD U-Net saved -> {save_dir}")
