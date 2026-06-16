"""Output-Grounding (OG) fine-tune of a CLIP text encoder against a FROZEN SD UNet.

The text-encoder (TE) family floor (Sph+OT 15.6, LSSE ~20 ASR; never reaching ODACE 4.0) comes
from optimizing a *text-embedding proxy* that underdetermines the image-level concept. This script
bridges TE methods toward ODACE's output-grounded depth WITHOUT training the UNet: it fine-tunes
ONLY the CLIP text encoder so that, for forget prompts, the FROZEN SD UNet's noise prediction
matches the unconditional (concept-absent) prediction, while a retain anchor keeps general prompts'
predictions unchanged. Cheaper than full ODACE (UNet stays frozen) but grounded in the actual
denoiser output rather than the embedding.

  L_forget = MSE( UNet(x_t, t, TE(c_forget)),  UNet(x_t, t, TE_frozen("")).detach() )   # erase
  L_retain = MSE( UNet(x_t, t, TE(c_retain)),  UNet(x_t, t, TE_frozen(c_retain)).detach() ) # keep
  L        = L_retain + og_eta * L_forget

It applies on TOP of ANY base text encoder dir (raw CLIP, Sph+OT final, LSSE final) and writes a
new CLIPTextModel checkpoint that eval/xeval.py loads via kind="te_swap" (B of the improvement plan:
the output-grounding hybrid). Random latents at random timesteps are used as x_t (no VAE/real images
needed) — a cheap ESD/ODACE-style objective over the latent distribution.
"""
from __future__ import annotations

import argparse, os, sys
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cost_utils import CostMeter  # noqa: E402  (env-aware training-cost logging)

SD14 = "CompVis/stable-diffusion-v1-4"


def read_prompts(path: str) -> list[str]:
    out = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    if not out:
        raise ValueError(f"no prompts in {path}")
    return out


def load_te(base_te: str, device, dtype=torch.float32) -> CLIPTextModel:
    """base_te == 'raw' -> SD14 text_encoder subfolder; else a saved CLIPTextModel dir."""
    if base_te == "raw":
        te = CLIPTextModel.from_pretrained(SD14, subfolder="text_encoder", torch_dtype=dtype)
    else:
        te = CLIPTextModel.from_pretrained(base_te, torch_dtype=dtype)
    return te.to(device)


def main():
    ap = argparse.ArgumentParser(description="Output-grounding TE fine-tune (frozen UNet)")
    ap.add_argument("--base_te", required=True,
                    help="'raw' or a CLIPTextModel dir to start from (e.g. Sph+OT/LSSE final)")
    ap.add_argument("--forget_file", required=True)
    ap.add_argument("--retain_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--og_eta", type=float, default=1.0, help="forget weight (ESD-style)")
    ap.add_argument("--neg_guidance", type=float, default=3.0,
                    help="ESD negative-guidance scale: forget target = uncond - g*(concept - uncond)")
    ap.add_argument("--t_lo", type=int, default=50, help="min sampled timestep (skip washed-out high noise)")
    ap.add_argument("--t_hi", type=int, default=800, help="max sampled timestep (conditioning-informative band)")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exp_name", type=str, default="og")
    args = ap.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- frozen SD UNet + scheduler; tokenizer ---
    # UNet in bf16 (inference-only target): halves VRAM/compute vs fp32 and avoids the fp32 cuBLAS
    # workspace OOM (CUBLAS_STATUS_EXECUTION_FAILED) that crashed the original fp32 backward. The
    # trainable TE stays fp32 for stable Adam; forward runs under bf16 autocast.
    pipe = StableDiffusionPipeline.from_pretrained(SD14, torch_dtype=torch.bfloat16, safety_checker=None)
    unet = pipe.unet.to(device).eval().requires_grad_(False)
    tokenizer: CLIPTokenizer = pipe.tokenizer
    noise_sched = DDPMScheduler.from_config(pipe.scheduler.config)
    n_train_t = noise_sched.config.num_train_timesteps
    latent_ch = unet.config.in_channels
    latent_res = 64  # 512/8

    # --- trainable TE (from base) + frozen TE (reference for retain/uncond targets) ---
    text_encoder = load_te(args.base_te, device, dtype=torch.float32).train().requires_grad_(True)
    frozen_te = load_te(args.base_te, device, dtype=torch.bfloat16).eval().requires_grad_(False)

    opt = torch.optim.Adam(text_encoder.parameters(), lr=args.lr)

    forget = read_prompts(args.forget_file)
    retain = read_prompts(args.retain_file)
    print(f"[og] base={args.base_te} forget={len(forget)} retain={len(retain)} "
          f"steps={args.steps} og_eta={args.og_eta} device={device}", flush=True)

    def tok(texts):
        return tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device)

    def emb(te, ids):
        return te(ids)[0]

    g = torch.Generator(device="cpu").manual_seed(args.seed)

    def sample_batch(pool, k):
        idx = torch.randint(0, len(pool), (k,), generator=g).tolist()
        return [pool[i] for i in idx]

    DT = torch.bfloat16
    t_hi = min(args.t_hi, n_train_t)
    with CostMeter(args.exp_name, str(out_dir), steps=args.steps):
        for step in range(1, args.steps + 1):
            opt.zero_grad(set_to_none=True)
            bf = sample_batch(forget, args.batch)
            br = sample_batch(retain, args.batch)
            ids_f, ids_r = tok(bf), tok(br)
            ids_uncond = tok([""] * args.batch)

            # shared random latent state x_t and timestep across cond/uncond for a fair contrast.
            # Sample t in an informative mid-band: at very high noise the conditioning is washed out
            # (the original full-range uniform t made the loss collapse to ~0 with no gradient).
            x = torch.randn(args.batch, latent_ch, latent_res, latent_res, device=device, dtype=DT)
            t = torch.randint(args.t_lo, t_hi, (args.batch,), device=device).long()

            # ESD-style negatively-guided forget target: push the forget prompt's prediction AWAY
            # from the concept (uncond - g*(concept - uncond)), giving a real gradient instead of the
            # signal-free "match uncond". Retain target = the frozen prediction (locality anchor).
            with torch.no_grad(), torch.autocast("cuda", dtype=DT):
                n_unc = unet(x, t, encoder_hidden_states=emb(frozen_te, ids_uncond)).sample
                n_for = unet(x, t, encoder_hidden_states=emb(frozen_te, ids_f)).sample
                tgt_forget = (n_unc - args.neg_guidance * (n_for - n_unc)).float()
                tgt_retain = unet(x, t, encoder_hidden_states=emb(frozen_te, ids_r)).sample.float()

            # forget and retain backprop SEPARATELY so only one full-UNet grad graph is alive at a
            # time -> halves peak activation memory (the fp32 two-graph backward was the OOM).
            with torch.autocast("cuda", dtype=DT):
                pred_forget = unet(x, t, encoder_hidden_states=emb(text_encoder, ids_f)).sample
                L_forget = F.mse_loss(pred_forget.float(), tgt_forget) * args.og_eta
            L_forget.backward()

            with torch.autocast("cuda", dtype=DT):
                pred_retain = unet(x, t, encoder_hidden_states=emb(text_encoder, ids_r)).sample
                L_retain = F.mse_loss(pred_retain.float(), tgt_retain)
            L_retain.backward()

            opt.step()

            if step % 25 == 0 or step == 1:
                print(f"[og] step {step}/{args.steps} L_forget={L_forget.item():.4f} "
                      f"L_retain={L_retain.item():.4f}", flush=True)

    final = out_dir / "final"
    text_encoder.eval().save_pretrained(str(final))
    tokenizer.save_pretrained(str(final))
    print(f"[og] DONE -> {final}", flush=True)


if __name__ == "__main__":
    main()
