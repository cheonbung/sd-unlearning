"""
LPIPS Evaluator for artistic style forgetting.

Paper defines (Table 2):
  LPIPS_f  – forgetting effectiveness: perceptual distance between
             target-style images generated BEFORE and AFTER forgetting.
             Higher = better forgetting.
  LPIPS_m  – style retention: distance between non-target-style images
             generated before and after forgetting.
             Lower = better retention.
  LPIPS_d  = LPIPS_f - LPIPS_m  (overall difference, higher = better)

Usage:
  evaluator = LPIPSEvaluator()
  result = evaluator.compare_directories(
      before_dir="outputs/sd_baseline/vangogh",
      after_dir="outputs/fcf_p_vangogh/final/vangogh",
  )
  print(result["lpips_mean"])
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_lpips():
    try:
        import lpips
        return lpips
    except ImportError:
        raise ImportError(
            "lpips package is required.\n"
            "Install with: pip install lpips"
        )


class LPIPSEvaluator:
    """
    Computes LPIPS perceptual distance between paired image directories.

    Args:
        net:    Backbone network ('alex', 'vgg', 'squeeze'). Paper uses default.
        device: torch.device
    """

    def __init__(self, net: str = "alex", device: torch.device = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.net = net
        self._loss_fn = None

    def _get_loss_fn(self):
        if self._loss_fn is None:
            lpips = _load_lpips()
            self._loss_fn = lpips.LPIPS(net=self.net).to(self.device)
            self._loss_fn.eval()
        return self._loss_fn

    def _load_image_tensor(self, path: str, size: int = 512) -> torch.Tensor:
        """Load image as a normalized tensor in [-1, 1], shape (1, 3, H, W)."""
        img = Image.open(path).convert("RGB").resize((size, size))
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0   # [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def compute_lpips(self, img_path_a: str, img_path_b: str) -> float:
        """Compute LPIPS distance between two images."""
        loss_fn = self._get_loss_fn()
        ta = self._load_image_tensor(img_path_a)
        tb = self._load_image_tensor(img_path_b)
        return loss_fn(ta, tb).item()

    @torch.no_grad()
    def compare_directories(
        self,
        before_dir: str,
        after_dir:  str,
        file_ext:   str = "*.png",
    ) -> Dict[str, float]:
        """
        Compute mean LPIPS between paired images in two directories.

        Images are paired by filename (sorted order).
        Returns mean LPIPS distance.
        """
        before_files = sorted(Path(before_dir).glob(file_ext))
        after_files  = sorted(Path(after_dir).glob(file_ext))

        if not before_files:
            logger.warning(f"No images found in: {before_dir}")
            return {"lpips_mean": 0.0, "lpips_std": 0.0, "n_pairs": 0}
        if not after_files:
            logger.warning(f"No images found in: {after_dir}")
            return {"lpips_mean": 0.0, "lpips_std": 0.0, "n_pairs": 0}

        n = min(len(before_files), len(after_files))
        distances = []
        for i in range(n):
            d = self.compute_lpips(str(before_files[i]), str(after_files[i]))
            distances.append(d)

        distances = np.array(distances)
        result = {
            "lpips_mean": float(distances.mean()),
            "lpips_std":  float(distances.std()),
            "n_pairs":    n,
        }
        logger.info(
            f"LPIPS | {Path(before_dir).name} ↔ {Path(after_dir).name} | "
            f"mean={result['lpips_mean']:.4f} ± {result['lpips_std']:.4f}  (n={n})"
        )
        return result

    def compute_style_forgetting_metrics(
        self,
        baseline_dir:    str,   # SD baseline images for target style
        forgetting_dir:  str,   # FCF images for target style  (LPIPS_f)
        baseline_other:  str,   # SD baseline images for non-target styles
        forgetting_other: str,  # FCF images for non-target styles   (LPIPS_m)
    ) -> Dict[str, float]:
        """
        Compute LPIPS_f, LPIPS_m, and LPIPS_d as defined in the paper.

        LPIPS_f  = compare(baseline_dir, forgetting_dir)    ↑ higher = better forget
        LPIPS_m  = compare(baseline_other, forgetting_other) ↓ lower  = better retain
        LPIPS_d  = LPIPS_f - LPIPS_m                        ↑ higher = better overall
        """
        res_f = self.compare_directories(baseline_dir,   forgetting_dir)
        res_m = self.compare_directories(baseline_other, forgetting_other)

        lpips_f = res_f["lpips_mean"]
        lpips_m = res_m["lpips_mean"]
        lpips_d = lpips_f - lpips_m

        result = {
            "LPIPS_f": lpips_f,
            "LPIPS_m": lpips_m,
            "LPIPS_d": lpips_d,
        }
        logger.info(
            f"Style Forgetting | LPIPS_f={lpips_f:.2f}  "
            f"LPIPS_m={lpips_m:.2f}  LPIPS_d={lpips_d:.2f}"
        )
        return result
