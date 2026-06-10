# fcf-novel-methods

본 논문(FCF, Fortified Concept Forgetting)의 메인 코드(`fcf/`)와 **완전히 독립적인** 실험용 서브 프로젝트입니다. 새로운 연구 아이디어(N1, N5, N6)를 본 코드 무결성을 깨뜨리지 않고 검증하기 위해 만들어졌습니다.

> **독립성 원칙**: 이 폴더 내부의 어떤 코드도 부모 프로젝트의 `fcf/` 패키지를 import하지 않습니다. 베이스 학습 코드(FCFTrainer, FCFDataset 등)는 모두 `core/`에 로컬 복사되어 있습니다.

---

## 1. 폴더 구조

```
models/novel/
├── README.md                  ← 본 문서
├── requirements.txt           ← 의존 패키지 (torch, transformers, ...)
├── train.py                   ← 학습 엔트리포인트
│
├── core/                      ← 베이스 FCF 코드의 로컬 독립 복사본
│   ├── __init__.py
│   ├── trainer.py             ← FCFTrainer (Stage 1 + FCF-P/FCF-E)
│   ├── dataset.py             ← FCFDataset (CSV / 텍스트 파일 로더)
│   └── noise_utils.py         ← 5자 무작위 노이즈 (논문 부록 사양)
│
├── methods/                   ← 새 연구 아이디어 (본 서브 프로젝트의 핵심)
│   ├── __init__.py
│   ├── spherical.py           ← N5: Riemannian 구면 측지선 사영
│   ├── ot_noise.py            ← N6: Optimal-Transport 기반 노이즈 학습
│   ├── cap_analyzer.py        ← N1: Causal Activation Patching (분석 전용)
│   └── trainer.py             ← NovelFCFTrainer (FCFTrainer 상속, N5/N6 연결)
│
├── configs/
│   └── nudity_v2.yaml         ← N5+N6 적용 학습 설정
│
├── scripts/
│   ├── run_cap_analysis.py    ← N1 CLI
│   └── learn_ot_noise.py      ← N6 CLI
│
└── tests/
    ├── conftest.py            ← Mock CLIP 모델 + 공유 fixture
    ├── test_spherical.py      ← N5 단위 테스트
    ├── test_ot_noise.py       ← N6 단위 테스트
    └── test_cap_analyzer.py   ← N1 단위 테스트
```

---

## 2. 구현된 연구 아이디어

### N5 — Riemannian Geodesic Forgetting (RG-FCF)

**파일**: `methods/spherical.py`, 연결: `methods/trainer.py`

CLIP 텍스트 임베딩은 임베딩 공간의 좁은 cone 위에 모여 있다는 사실(Ethayarajh, EMNLP 2019)에서 출발합니다. 논문의 FCF-P는 유클리드 사영을 사용하므로 cleaned target이 이 cone **밖으로** 밀려나 UNet에 OOD 입력이 될 수 있습니다.

대안으로 **단위 초구(unit hypersphere) 위에서 측지선(geodesic)을 따라 한 걸음만 이동**합니다.

1. `target_mean`, `concept_mean`을 단위 벡터로 정규화 (S^{L·D-1} 위의 점)
2. `target` 지점에서 `concept` 쪽을 향하는 접선 벡터(log map) 계산
3. 그 접선을 따라 `-eta_clean` 방향으로 한 걸음 이동
4. exp map으로 다시 구면 위로 사영
5. 원래 `target_mean`의 크기로 재스케일링 → UNet 입력 분포 보존

### N6 — Optimal-Transport-derived Noise Prompts (OT-FCF)

**파일**: `methods/ot_noise.py`, CLI: `scripts/learn_ot_noise.py`

논문은 "5글자 무작위 문자열"을 노이즈 P_n으로 사용하지만, 이것이 임베딩 공간에서 **최적의 망각 도착점**이라는 보장이 없습니다.

해결: 후보 노이즈 문자열들 중 explicit 프롬프트 분포에서 **Wasserstein 거리가 최대**가 되는 부분집합을 선택합니다.

알고리즘 (`OTNoiseLearner.learn`):

1. 논문 규칙(5글자, 심볼+알파벳+숫자, 중복 없음)으로 `n_candidates`개의 후보 노이즈 생성
2. 동결된 CLIP 텍스트 인코더로 후보·explicit 모두 임베딩
3. 각 후보를 explicit 분포로부터의 sliced Wasserstein 거리 (또는 Sinkhorn divergence)로 점수화
4. 가장 거리가 큰 상위 `n_select`개 선택
5. 결과 + 메타데이터를 JSON으로 저장

POT 라이브러리를 일부러 사용하지 않고, sliced Wasserstein과 Sinkhorn iteration을 직접 구현해 외부 의존성을 0으로 유지했습니다.

### N1 — Causal Activation Patching (CAP) 분석 전용

**파일**: `methods/cap_analyzer.py`, CLI: `scripts/run_cap_analysis.py`

ROME(Meng et al., NeurIPS 2022) / Causal Mediation(Vig et al., NeurIPS 2020) 방법론으로 **어느 layer가 특정 개념을 인코딩하는지** 인과적으로 찾습니다.

알고리즘 (`CausalActivationPatcher.analyze`):

각 layer L에 대해
1. **noise** 프롬프트로 forward → 모든 layer의 출력 캐싱
2. **clean** (explicit) 프롬프트로 forward → 패치 없는 출력 저장
3. **patched** forward: layer L의 출력을 캐싱한 noise activation으로 교체, 하위 layer는 정상 진행
4. `score(L) = ||clean_output - patched_output||` (배치 평균)

`score`가 큰 layer가 해당 개념과 **인과적으로 강하게 연관**되어 있습니다. 결과는 JSON heatmap으로 저장되어 후속 ROME-style 편집의 타겟 layer를 결정할 수 있습니다(편집 단계는 의도적으로 다음 실험으로 분리).

---

## 3. 사용 방법

### 3.1 의존 패키지 설치

```powershell
# fcf conda 환경에 추가 패키지 설치
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" -m pip install -r fcf-novel-methods\requirements.txt
```

### 3.2 N5 (Riemannian) 학습 실행

```powershell
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" fcf-novel-methods\train.py --config fcf-novel-methods\configs\nudity_v2.yaml --manifold spherical
```

- `--manifold euclidean`이면 논문 동작과 동일(검증용 회귀).
- 결과는 `models/novel/outputs/fcf_p_v2_nudity/` 에 저장 → 부모 `outputs/` 무영향.

### 3.3 N6 (OT 노이즈) 어휘 학습 + 학습에 사용

```powershell
# Step 1: OT 어휘 학습 (frozen CLIP만 사용, GPU 권장)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" fcf-novel-methods\scripts\learn_ot_noise.py --explicit_file data\prompts\nudity_explicit.txt --output fcf-novel-methods\outputs\ot_noise\nudity_learned.json

# Step 2: 학습된 어휘를 사용해 FCF 학습
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" fcf-novel-methods\train.py --config fcf-novel-methods\configs\nudity_v2.yaml --ot_noise_file fcf-novel-methods\outputs\ot_noise\nudity_learned.json
```

### 3.4 N1 (CAP) 인과 분석 실행

```powershell
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" fcf-novel-methods\scripts\run_cap_analysis.py --concept nudity --explicit_file data\prompts\nudity_explicit.txt --output fcf-novel-methods\outputs\cap\nudity_heatmap.json
```

### 3.5 단위 테스트 실행

```powershell
# 본 서브 프로젝트만 테스트
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" -m pytest tests\ -v
```

---

## 4. 본 프로젝트(`fcf/`)와의 관계

| 항목 | 본 프로젝트 (`fcf/`, `train_fcf.py`) | 본 서브 프로젝트 (`models/novel/`) |
|---|---|---|
| 알고리즘 | 논문 그대로 (FCF-P, FCF-E) | N1 분석 + N5 (구면) + N6 (OT 노이즈) |
| 학습 코드 | 정본 — 수정 금지 | 로컬 독립 복사본 (core/) |
| 하이퍼파라미터 | `CLAUDE.md`에 잠금 (lr=2.5e-5, η=0.25, μ_p=0.7, μ_e=1.0) | 동일 기본값 유지, config로 override 가능 |
| 출력 | `outputs/` | `models/novel/outputs/` |
| import 의존성 | — | **`fcf/`를 import하지 않음** |

본 코드의 무결성을 위해 다음을 **절대 하지 않습니다**:
- `fcf/`, `train_fcf.py`, `evaluate.py`, `configs/*.yaml`, `tests/`, `evaluation/`, `scripts/`를 수정
- 본 프로젝트 `outputs/`에 결과 저장
- CLIP 외 모듈(UNet, VAE) 학습

---

## 5. 검증 체크리스트

본 서브 프로젝트의 정합성은 다음으로 검증됩니다.

- [x] 단위 테스트: `pytest models/novel/tests/ -v` 전부 통과
- [x] `manifold='euclidean'` 옵션은 본 프로젝트 FCF-P와 수치적으로 동일 (`test_euclidean_matches_fcf_p_formula`)
- [x] `OTNoiseResult` JSON save/load round-trip (`test_ot_result_round_trip`)
- [x] CAP analyzer가 mock CLIP에서 모든 layer를 식별 (`test_get_encoder_layers_finds_layers`)
- [x] `fcf/` 패키지 import가 코드 내 단 한 곳도 없음 (`grep -r "from fcf" models/novel/` 결과 비어 있음)

---

## 6. 참고 문헌

- **N1**: Vig et al., "Causal Mediation Analysis", NeurIPS 2020 / Meng et al., "Locating and Editing Factual Associations in GPT" (ROME), NeurIPS 2022
- **N5**: Ethayarajh, "How Contextual are Contextualized Word Representations?", EMNLP 2019 (CLIP cone 현상의 동기)
- **N6**: Cuturi, "Sinkhorn Distances", NIPS 2013 / Bonneel et al., "Sliced and Radon Wasserstein Barycenters", JMIV 2015
- **본 논문**: Fan et al., "Fortified Concept Forgetting for text-to-image generative models by machine unlearning on CLIP", Computer Standards & Interfaces 97 (2026) 104142
