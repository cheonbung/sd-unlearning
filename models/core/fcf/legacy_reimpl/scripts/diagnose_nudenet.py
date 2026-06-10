"""
NudeNet 진단 스크립트.
샘플 이미지에서 NudeNet이 실제로 어떤 레이블/스코어를 반환하는지 확인.
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DIRS_TO_CHECK = {
    "baseline_i2p":      BASE_DIR / "outputs/eval_baseline_nudity/images/fcf_nudity/i2p",
    "fcf_p_i2p":         BASE_DIR / "outputs/eval/images/fcf_nudity/i2p",
    "baseline_rab":      BASE_DIR / "outputs/eval_baseline_nudity/images/fcf_nudity/ring_a_bell",
}

N_SAMPLES = 5  # 디렉토리당 샘플 수

def main():
    from nudenet import NudeDetector
    detector = NudeDetector()
    print("NudeNet loaded.\n")

    for label, img_dir in DIRS_TO_CHECK.items():
        if not img_dir.exists():
            print(f"[SKIP] {label}: 디렉토리 없음 ({img_dir})\n")
            continue

        imgs = sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
        print(f"=== {label} ({len(imgs)} images) ===")

        for img in imgs[:N_SAMPLES]:
            results = detector.detect(str(img))
            print(f"  {img.name}:")
            if not results:
                print("    (감지 없음 — NudeNet 반환값 빈 리스트)")
            else:
                for r in results:
                    print(f"    {r.get('class','?'):35s}  score={r.get('score', 0):.4f}")
        print()

if __name__ == "__main__":
    main()
