"""
FID and CLIP Score Evaluator.

Paper Table 3: evaluates generative quality of the nudity-forgotten model
on COCO-30K (safe subset) using:
  - FID  (Fréchet Inception Distance) – lower is better
  - CLIP Score                         – higher is better

These metrics check that non-target generation is not harmed by concept forgetting.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)


class FIDCLIPEvaluator:
    """
    Computes FID and CLIP Score between generated images and reference prompts.

    FID is computed between:
      - a reference real image directory (e.g. COCO-30K subset)
      - the generated image directory

    CLIP Score is computed between each generated image and its prompt.

    Args:
        device: torch.device
        clip_model_id: CLIP model for CLIP score computation
    """

    def __init__(
        self,
        device: torch.device = None,
        clip_model_id: str = "openai/clip-vit-large-patch14",
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.clip_model_id = clip_model_id
        self._clip_model = None
        self._clip_processor = None

    # ------------------------------------------------------------------ #
    #  CLIP Score                                                          #
    # ------------------------------------------------------------------ #

    def _load_clip(self):
        if self._clip_model is None:
            from transformers import CLIPModel, CLIPProcessor
            logger.info(f"Loading CLIP for scoring: {self.clip_model_id}")
            self._clip_processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            self._clip_model = CLIPModel.from_pretrained(self.clip_model_id).to(self.device)
            self._clip_model.eval()

    @torch.no_grad()
    def compute_clip_score(
        self,
        image_dir:  str,
        prompts:    List[str],
        batch_size: int = 32,
    ) -> float:
        """
        Compute mean CLIP score (cosine similarity between image and text).

        Args:
            image_dir:  Directory containing generated images (sorted order).
            prompts:    Corresponding text prompts.
            batch_size: Images per CLIP forward pass (default 32).

        Returns:
            Mean CLIP score (scaled ×100 to match paper convention).
        """
        self._load_clip()

        image_files = sorted(
            list(Path(image_dir).glob("*.png")) +
            list(Path(image_dir).glob("*.jpg"))
        )
        n = min(len(image_files), len(prompts))
        if n == 0:
            logger.warning("No images or prompts for CLIP score computation.")
            return 0.0

        scores = []
        pbar = tqdm(range(0, n, batch_size), desc="CLIP Score", leave=False)
        for s in pbar:
            e = min(s + batch_size, n)
            batch_images  = [Image.open(str(image_files[i])).convert("RGB") for i in range(s, e)]
            batch_prompts = prompts[s:e]

            inputs = self._clip_processor(
                text=batch_prompts,
                images=batch_images,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                outputs = self._clip_model(**inputs)

            img_emb = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
            # 배치 내 각 이미지-텍스트 쌍의 코사인 유사도
            sims = (img_emb * txt_emb).sum(dim=-1) * 100.0   # (B,)
            scores.extend(sims.cpu().tolist())

        mean_score = float(np.mean(scores))
        logger.info(f"CLIP Score: {mean_score:.2f}  (n={n})")
        return mean_score

    # ------------------------------------------------------------------ #
    #  FID                                                                 #
    # ------------------------------------------------------------------ #

    def compute_fid(
        self,
        generated_dir: str,
        reference_dir: str,
        batch_size:    int = 32,
        num_workers:   int = 0,
    ) -> float:
        """
        Compute FID between generated images and reference images.

        Requires the clean-fid package: pip install clean-fid

        Args:
            generated_dir: Directory with generated images.
            reference_dir: Directory with real reference images (e.g. COCO-30K).

        Returns:
            FID score (lower is better).
        """
        try:
            from cleanfid import fid as clean_fid
        except ImportError:
            raise ImportError(
                "clean-fid is required for FID computation.\n"
                "Install with: pip install clean-fid"
            )

        logger.info(f"Computing FID: {generated_dir} ↔ {reference_dir}")
        # cleanfid's DataParallel hardcodes device_ids=[0,1,...] with output on device_ids[0].
        # On non-zero GPUs this causes a device mismatch error.
        # use_dataparallel=False bypasses DataParallel entirely, allowing any GPU index.
        score = clean_fid.compute_fid(
            fdir1=generated_dir,
            fdir2=reference_dir,
            device=self.device,
            batch_size=batch_size,
            num_workers=num_workers,
            use_dataparallel=False,
        )
        logger.info(f"FID: {score:.2f}")
        return score

    # ------------------------------------------------------------------ #
    #  Combined evaluation                                                 #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        image_dir:     str,
        prompts:       List[str],
        reference_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compute both FID (if reference_dir provided) and CLIP score.

        Returns:
            dict with 'fid' and 'clip_score' keys.
        """
        result: Dict[str, float] = {}

        clip_score = self.compute_clip_score(image_dir, prompts)
        result["clip_score"] = clip_score

        if reference_dir is not None:
            fid = self.compute_fid(image_dir, reference_dir)
            result["fid"] = fid
        else:
            logger.info("No reference_dir provided – skipping FID computation.")

        return result
