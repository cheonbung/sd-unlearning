# fcf-novel-methods — Claude 컨텍스트 파일

마지막 업데이트: 2026-05-26

---

## 이 프로젝트가 무엇인가

부모 프로젝트 (`fcf/`, `train_fcf.py`) **본 논문 구현체와 완전히 독립적인 실험용 서브 프로젝트**. 새로운 연구 아이디어 (N1/N5/N6) 를 본 코드 무결성을 깨지 않고 검증.

상세 구조/철학은 [README.md](README.md) 참조. 실험 결과는 [`@docs/results.md`](docs/results.md).

---

## 구현된 연구 아이디어

| ID | 이름 | 핵심 파일 | 역할 |
|---|---|---|---|
| **N1** | Causal Activation Patching (CAP) | [`methods/cap_analyzer.py`](methods/cap_analyzer.py) | layer-wise 인과 분석 (편집 X, 분석만) |
| **N5** | Riemannian Geodesic FCF (RG-FCF) | [`methods/spherical.py`](methods/spherical.py), [`methods/trainer.py`](methods/trainer.py) | CLIP 임베딩 hypersphere 위 geodesic 사영 |
| **N6** | OT-derived Noise Vocabulary (OT-FCF) | [`methods/ot_noise.py`](methods/ot_noise.py), [`scripts/learn_ot_noise.py`](scripts/learn_ot_noise.py) | Sliced Wasserstein 으로 noise vocabulary 최적화 |

---

## 핵심 하이퍼파라미터 (부모 프로젝트와 동일 — 변경 금지)

| 파라미터 | 값 | 비고 |
|---|---|---|
| `learning_rate` | 2.5e-5 | CLIP 인코더 학습률 |
| `eta` (η) | 0.25 | Stage 2 이동 강도 |
| `mu_p` (μ_p) | 0.7 | FCF-P 임계값 |
| `mu_e` (μ_e) | 1.0 | FCF-E 임계값 |
| `num_epochs` | 60 | Stage 1 + Stage 2 동일 |
| `seed` | 42 | 재현성 |

설정 파일: [`configs/nudity_v2.yaml`](configs/nudity_v2.yaml) (Windows 학습용), [`configs/nudity_v2_wsl.yaml`](configs/nudity_v2_wsl.yaml) (WSL FID 평가용 — `coco_dir`만 `/mnt/c/...`).

---

## 절대 변경 금지 (Code Sovereignty)

부모 프로젝트 무결성을 위해 **다음 파일/디렉토리는 절대 수정하지 않는다**:

- `fcf/`, `train_fcf.py`, `evaluate.py`
- 부모 `configs/*.yaml`, `tests/`, `evaluation/`, `scripts/`
- 부모 `outputs/` (이 서브프로젝트 결과는 `fcf-novel-methods/outputs/`에만 저장)

`fcf/` 패키지 import 금지 — 베이스 학습 코드는 모두 [`core/`](core/)에 로컬 복사본 존재.

---

## 알고리즘 구현 규칙 (디버깅으로 확인된 사항)

### N5 (Spherical)

- `manifold='euclidean'`은 부모 FCF-P와 **수치적으로 동일**해야 함 (`test_euclidean_matches_fcf_p_formula` 보장).
- Spherical: log/exp map 후 **원래 `target_mean`의 크기로 재스케일**하여 UNet 입력 분포 보존.

### N6 (OT Noise)

- 후보 생성은 **부모 `core/noise_utils.py` 규칙** 준수 (5글자, 심볼+알파벳+숫자, 중복 없음).
- POT 라이브러리 의존 금지 — sliced Wasserstein / Sinkhorn 직접 구현.
- 저장 포맷: `OTNoiseResult` JSON (round-trip 보장, `test_ot_result_round_trip`).

### N1 (CAP) — **버그 수정 완료 (2026-05-22)**

- **이전 버그**: 레이어 i의 출력 전체를 noise activation으로 교체하면, 후속 레이어가 결정론적으로 같은 endpoint로 수렴 → 12 레이어 모두 동일 점수 (0.7884).
- **수정**: 단일 토큰 위치(`patch_token_idx=1`)만 패치 → attention mixing이 레이어별로 차별화 → 레이어 0이 최고 (0.4952), 레이어 11이 최저 (0.0529).
- CAP은 **분석 전용** — ROME-style 편집 단계는 의도적으로 후속 연구로 분리.

---

## 자주 쓰는 커맨드

> **PowerShell 환경 — `\` 줄 연속 금지, 한 줄 작성**. WSL 사용 시 path quote 필수 (괄호/공백 포함).

### 학습 (PowerShell, Windows)

```powershell
# N5 spherical
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train.py --config configs\nudity_v2.yaml --manifold spherical

# N5 euclidean (baseline 회귀 검증)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train.py --config configs\nudity_v2.yaml --manifold euclidean

# N6 OT vocabulary 학습 (Step 1)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" scripts\learn_ot_noise.py --explicit_file data\prompts\nudity_explicit.txt --output outputs\ot_noise\nudity_learned.json

# N5 + N6 결합 학습 (Step 2)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train.py --config configs\nudity_v2.yaml --manifold spherical --ot_noise_file outputs\ot_noise\nudity_learned.json
```

### 학습 (WSL bash, GPU)

```bash
cd "/mnt/c/Users/vip/Desktop/byeongcheon(server_pc)/SD_unlearning/fcf-novel-methods"
source /root/anaconda3/etc/profile.d/conda.sh && conda activate fcf-novel
python train.py --config configs/nudity_v2.yaml --manifold spherical --ot_noise_file outputs/ot_noise/nudity_learned.json
```

### N1 CAP 분석

```powershell
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" scripts\run_cap_analysis.py --concept nudity --explicit_file data\prompts\nudity_explicit.txt --output outputs\cap\nudity_heatmap.json --n_samples 8
```

### 평가 (ASR + Quality)

```bash
# ASR
python evaluate.py --encoder_dir outputs/fcf_p_v2_nudity_spherical_ot/final --concept nudity --eval_type asr --output_dir outputs/eval/spherical_ot_asr

# Quality + FID (WSL config 필수 — coco_dir이 /mnt/c/... 로 설정됨)
python evaluate.py --config configs/nudity_v2_wsl.yaml --encoder_dir outputs/fcf_p_v2_nudity_spherical_ot/final --concept nudity --eval_type quality --output_dir outputs/eval/spherical_ot_quality
```

### 단위 테스트

```powershell
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" -m pytest tests\ -v
```

---

## 디버깅으로 확인된 함정

### 1. `ot_noise_file` 경로 — double-path 버그

- `train.py`의 `build_dataset`은 `cfg["ot_noise_file"]`을 `base_dir = fcf-novel-methods/` 기준으로 prepend.
- ❌ `--ot_noise_file fcf-novel-methods/outputs/ot_noise/nudity_learned.json` → `fcf-novel-methods/fcf-novel-methods/...` 이중 경로 에러
- ✅ `--ot_noise_file outputs/ot_noise/nudity_learned.json` (서브프로젝트 상대 경로)

### 2. `run_cap_analysis.py` `--output` 경로

- 동일 문제: `_PROJECT_ROOT = fcf-novel-methods/`에 prepend.
- ✅ `--output outputs/cap/nudity_heatmap.json` (서브프로젝트 상대)

### 3. FID 평가 시 `coco_dir` 경로

- 부모 `configs/nudity_v2.yaml`은 Windows 경로 (`C:/Users/...`) → WSL Python이 못 읽음.
- ✅ WSL 평가 시 [`configs/nudity_v2_wsl.yaml`](configs/nudity_v2_wsl.yaml) 사용 (`/mnt/c/...`).

### 4. 학습 출력 디렉토리 충돌

- 모든 학습이 `cfg["output_dir"] = "outputs/fcf_p_v2_nudity"`로 저장 → 후속 학습이 덮어씀.
- ✅ 학습 직후 `mv outputs/fcf_p_v2_nudity outputs/fcf_p_v2_nudity_<variant>` 로 변형별 분리.

### 5. WSL 환경 (Linux)

- conda env 이름: `fcf-novel` (부모는 `fcf`).
- `transformers 4.44.2` 고정 (5.x는 `CLIPTextModel` import 깨짐).
- `diffusers 0.30.3` 고정 (0.38.0은 torch 2.4.1과 `infer_schema` 비호환).
- nudity ASR 백엔드: `nudenet` v3 (`onnxruntime 1.23.2`), `cleanfid` for FID.

---

## 실험 결과 (수치)

[`@docs/results.md`](docs/results.md) — fcf-novel-methods 섹션 (Nudity, NudeNet v3, 50장 ASR + COCO ref FID + CAP 레이어 점수).

핵심 발견:
- **N5+N6 (Spherical+OT)** 가 5개 공격 모두에서 최저 ASR
- **N6 OT noise**가 적응형 공격 Ring-A-Bell(Re) 강건성에 결정적 (93.9% → 20%)
- **CAP**: 초기 레이어 (0–3) 가 nudity 개념 인코딩에 가장 큰 인과적 기여

---

## 코드 수정 체크리스트

1. 부모 `fcf/`, `train_fcf.py`, `evaluate.py`를 수정하지 않는가?
2. `fcf/` 패키지를 import하지 않는가? (`grep -r "from fcf" fcf-novel-methods/` 결과 비어 있어야 함)
3. 결과를 `fcf-novel-methods/outputs/`에만 저장하는가?
4. 단위 테스트 (`pytest fcf-novel-methods/tests/`) 전부 통과하는가?
5. 핵심 하이퍼파라미터 (lr, η, μ_p, μ_e) 를 변경하지 않았는가?
