"""
validate.py — FCF 프로젝트 하네스 검증 스크립트

코드 수정 후, 또는 Claude에게 작업을 시키기 전/후에 실행.
자동으로 잡아낼 수 있는 실수들을 빠르게 체크한다.

Usage:
    python scripts/validate.py           # 전체 검사
    python scripts/validate.py --quick   # 빠른 검사만 (import 제외)
"""

import ast
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = []
warnings = []


def check(condition: bool, fail_msg: str, warn_only: bool = False):
    if condition:
        print(f"  {PASS} {fail_msg.split('→')[0].strip()}")
    else:
        if warn_only:
            print(f"  {WARN} {fail_msg}")
            warnings.append(fail_msg)
        else:
            print(f"  {FAIL} {fail_msg}")
            errors.append(fail_msg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. 필수 파일 존재 확인
# ─────────────────────────────────────────────────────────────────────────────
def check_required_files():
    print("\n[1] 필수 파일 존재 확인")
    required = [
        "CLAUDE.md",
        "train_fcf.py",
        "evaluate.py",
        "generate_images.py",
        "requirements.txt",
        "fcf/__init__.py",
        "fcf/trainer.py",
        "fcf/dataset.py",
        "fcf/noise_utils.py",
        "configs/nudity_fcf_p.yaml",
        "configs/nudity_fcf_e.yaml",
        "configs/vangogh_fcf_p.yaml",
        "configs/vangogh_fcf_e.yaml",
        "configs/violence_fcf_p.yaml",
        "configs/violence_fcf_e.yaml",
    ]
    for f in required:
        check((ROOT / f).exists(), f"필수 파일 없음: {f}")

    # 평가 데이터 — 없으면 경고만 (외부에서 다운로드 필요)
    eval_data = [
        "data/eval/i2p_nudity.txt",
        "data/eval/i2p_violence.txt",
        "data/eval/ring_a_bell_nudity.txt",
        "data/eval/p4d_nudity.txt",
        "data/eval/unlearnDiffAtk_nudity.txt",
        "data/eval/vangogh_style.txt",
        "evaluation/q16_weights/prompts.p",
    ]
    for f in eval_data:
        check((ROOT / f).exists(), f"평가 데이터 없음 (다운로드 필요): {f}", warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. 논문 구현 규칙 위반 검사
# ─────────────────────────────────────────────────────────────────────────────
def check_implementation_rules():
    print("\n[2] 논문 구현 규칙 검사")

    trainer_path = ROOT / "fcf" / "trainer.py"
    if not trainer_path.exists():
        print(f"  {WARN} trainer.py 없음 — 건너뜀")
        return

    src = trainer_path.read_text(encoding="utf-8")

    # UNet에 gradient가 흐르면 안 됨
    has_unet_grad = (
        "unet.train()" in src or
        ("unet" in src and "requires_grad_(True)" in src)
    )
    check(not has_unet_grad,
          "UNet에 gradient가 설정됨 → CLIP 텍스트 인코더만 학습해야 함")

    # Stage 2 target은 루프 밖에서 계산돼야 함
    # 단순 휴리스틱: target 계산이 for/while 블록 안에 있으면 경고
    lines = src.splitlines()
    in_loop = False
    stage2_target_in_loop = False
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(("for ", "while ")):
            in_loop = True
        if in_loop and "stage2_target" in line and "=" in line:
            # 할당이 루프 안에 있으면 경고
            indent = len(line) - len(line.lstrip())
            if indent > 0:
                stage2_target_in_loop = True
    check(not stage2_target_in_loop,
          "Stage 2 target이 루프 내부에서 재계산되는 것으로 보임 → 루프 전에 1회만 계산해야 함",
          warn_only=True)

    # noise_utils는 dataset.py를 통해 간접 사용 — dataset import만 확인
    dataset_path = ROOT / "fcf" / "dataset.py"
    if dataset_path.exists():
        dataset_src = dataset_path.read_text(encoding="utf-8")
        check("noise_utils" in dataset_src,
              "dataset.py가 noise_utils를 import하지 않음 — 노이즈 텍스트 생성 누락 가능성",
              warn_only=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. config 파일 핵심 하이퍼파라미터 확인
# ─────────────────────────────────────────────────────────────────────────────
def check_configs():
    print("\n[3] Config 하이퍼파라미터 검사")
    import yaml  # noqa

    config_checks = {
        "configs/nudity_fcf_p.yaml": {"learning_rate": 2.5e-5, "eta": 0.25, "mu_p": 0.7},
        "configs/nudity_fcf_e.yaml": {"learning_rate": 2.5e-5, "eta": 0.25, "mu_e": 1.0},
    }

    for config_path, expected in config_checks.items():
        full_path = ROOT / config_path
        if not full_path.exists():
            print(f"  {WARN} {config_path} 없음 — 건너뜀")
            continue
        try:
            with open(full_path) as f:
                cfg = yaml.safe_load(f)
            for key, val in expected.items():
                actual = cfg.get(key)
                if actual is None:
                    check(False, f"{config_path}: '{key}' 키 없음", warn_only=True)
                else:
                    close = abs(float(actual) - float(val)) < 1e-8
                    check(close,
                          f"{config_path}: '{key}' = {actual} (논문 기준: {val})",
                          warn_only=True)
        except Exception as e:
            print(f"  {WARN} {config_path} 파싱 실패: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Python 문법 오류 검사
# ─────────────────────────────────────────────────────────────────────────────
def check_syntax():
    print("\n[4] Python 문법 검사")
    py_files = list(ROOT.glob("**/*.py"))
    py_files = [f for f in py_files if ".git" not in str(f) and "scripts/validate" not in str(f)]

    syntax_ok = True
    for py_file in py_files:
        try:
            ast.parse(py_file.read_text(encoding="utf-8"))
        except SyntaxError as e:
            check(False, f"문법 오류: {py_file.relative_to(ROOT)} line {e.lineno}: {e.msg}")
            syntax_ok = False
    if syntax_ok:
        print(f"  {PASS} {len(py_files)}개 파일 모두 문법 정상")


# ─────────────────────────────────────────────────────────────────────────────
# 5. outputs/ 구조 확인 (선택)
# ─────────────────────────────────────────────────────────────────────────────
def check_outputs():
    print("\n[5] Outputs 폴더 상태")
    outputs = ROOT / "outputs"
    if not outputs.exists():
        print(f"  {INFO} outputs/ 폴더 없음 (아직 학습 안 함)")
        return
    runs = [d for d in outputs.iterdir() if d.is_dir()]
    print(f"  {INFO} 저장된 실험: {len(runs)}개")
    for run in runs:
        checkpoints = list(run.glob("**/pytorch_model.bin")) + list(run.glob("**/*.safetensors"))
        print(f"    • {run.name}: 체크포인트 {len(checkpoints)}개")


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="문법 검사 건너뜀")
    args = parser.parse_args()

    print("=" * 55)
    print("  FCF 하네스 검증")
    print("=" * 55)

    check_required_files()
    check_implementation_rules()
    check_configs()
    if not args.quick:
        check_syntax()
    check_outputs()

    print("\n" + "=" * 55)
    if errors:
        print(f"  결과: {FAIL}  오류 {len(errors)}개, 경고 {len(warnings)}개")
        for e in errors:
            print(f"    ✗ {e}")
        sys.exit(1)
    elif warnings:
        print(f"  결과: {WARN}  오류 없음, 경고 {len(warnings)}개")
        for w in warnings:
            print(f"    ! {w}")
    else:
        print(f"  결과: {PASS}  모두 정상")
    print("=" * 55)


if __name__ == "__main__":
    main()
