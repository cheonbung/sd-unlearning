"""Safe Latent Diffusion (Schramowski et al., CVPR 2023) -- training-free reimplementation.

The diffusers 0.38 build here ships only a dummy `StableDiffusionPipelineSafe` stub (the
`stable_diffusion_safe` submodule is gone), so we reimplement the safety guidance directly from
the paper's Eqs. 3-8. Per denoising step we compute THREE noise predictions -- unconditional,
prompt-conditioned, and safety-concept-conditioned -- and steer the trajectory away from the
safety concept with a thresholded, momentum-accelerated guidance term:

    noise_guidance        = e_text - e_uncond                              # standard CFG dir
    scale                 = clamp(|e_text - e_safety| * s_S, max=1)
    safety_scale          = where(e_text - e_safety >= lambda, 0, scale)   # only where not already far
    g_safety              = (e_safety - e_uncond) * safety_scale + s_m * momentum
    momentum             := beta_m * momentum + (1 - beta_m) * g_safety
    if step >= warmup:    noise_guidance -= g_safety
    e_pred                = e_uncond + guidance_scale * noise_guidance

This matches the reference diffusers SLD implementation and the paper's four presets exactly
(delta, s_S, lambda, s_m, beta_m). Training-free: only the inference loop changes.

Used by eval/xeval.py (kind="sld") for ASR and by eval/eval_coco.py for COCO generation.
"""
from __future__ import annotations

import torch

# Paper Tab. (Sec. 4) / diffusers SafetyConfig -- (warmup delta, guidance s_S, threshold lambda,
# momentum s_m, mom_beta beta_m). Identical to Schramowski et al. Table.
SLD_CONFIGS = {
    "weak":   {"warmup": 15, "guidance_scale": 200,  "threshold": 0.0,   "momentum_scale": 0.0, "mom_beta": 0.0},
    "medium": {"warmup": 10, "guidance_scale": 1000, "threshold": 0.01,  "momentum_scale": 0.3, "mom_beta": 0.4},
    "strong": {"warmup": 7,  "guidance_scale": 2000, "threshold": 0.025, "momentum_scale": 0.5, "mom_beta": 0.7},
    "max":    {"warmup": 0,  "guidance_scale": 5000, "threshold": 1.0,   "momentum_scale": 0.5, "mom_beta": 0.7},
}

# Default unsafe/safety concept (paper Sec. 4 "(Un)safe Concept"; also used by Safe-CLIP).
SAFETY_CONCEPT = ("hate, harassment, violence, suffering, humiliation, harm, suicide, sexual, "
                  "nudity, bodily fluids, blood, obscene gestures, illegal activity, drug use, "
                  "theft, vandalism, weapons, child abuse, brutality, cruelty")


def _encode(pipe, prompt: str, device) -> torch.Tensor:
    tok = pipe.tokenizer([prompt], padding="max_length",
                         max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").to(device)
    return pipe.text_encoder(tok.input_ids)[0]


@torch.no_grad()
def sld_generate(pipe, prompt: str, cfg: dict, generator, steps: int, guidance_scale: float,
                 height: int = 512, width: int = 512, safety_concept: str = SAFETY_CONCEPT):
    """Generate one PIL image for `prompt` with SLD safety guidance preset `cfg`.

    Mirrors xeval.generate's contract (deterministic per-prompt latent, same scheduler) but runs
    the 3-way safety-guided denoise loop instead of pipe.__call__.
    """
    device = pipe.device
    dtype = pipe.unet.dtype

    emb_uncond = _encode(pipe, "", device)
    emb_text = _encode(pipe, prompt, device)
    emb_safety = _encode(pipe, safety_concept, device)
    text_embeddings = torch.cat([emb_uncond, emb_text, emb_safety]).to(dtype)

    scheduler = pipe.scheduler
    scheduler.set_timesteps(steps, device=device)
    latents = torch.randn((1, pipe.unet.config.in_channels, height // 8, width // 8),
                          generator=generator, device=device, dtype=dtype)
    latents = latents * scheduler.init_noise_sigma

    warmup = cfg["warmup"]
    s_S = cfg["guidance_scale"]
    threshold = cfg["threshold"]
    s_m = cfg["momentum_scale"]
    beta_m = cfg["mom_beta"]
    momentum = torch.zeros_like(latents)

    for i, t in enumerate(scheduler.timesteps):
        latent_in = scheduler.scale_model_input(torch.cat([latents] * 3), t)
        noise = pipe.unet(latent_in, t, encoder_hidden_states=text_embeddings).sample
        e_uncond, e_text, e_safety = noise.chunk(3)

        noise_guidance = e_text - e_uncond
        scale = torch.clamp(torch.abs(e_text - e_safety) * s_S, max=1.0)
        safety_scale = torch.where(
            (e_text - e_safety) >= threshold, torch.zeros_like(scale), scale)
        g_safety = (e_safety - e_uncond) * safety_scale + s_m * momentum
        momentum = beta_m * momentum + (1.0 - beta_m) * g_safety
        if i >= warmup:
            noise_guidance = noise_guidance - g_safety

        noise_pred = e_uncond + guidance_scale * noise_guidance
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    latents = latents / pipe.vae.config.scaling_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype)).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = (image[0].permute(1, 2, 0).float().cpu().numpy() * 255).round().astype("uint8")
    from PIL import Image
    return Image.fromarray(image)
