# LSSE — Claude 컨텍스트 파일

마지막 업데이트: 2026-05-27

---

## 이 프로젝트가 무엇인가

**Layer-Selective Semantic Erasure (LSSE)** — FCF 및 fcf-novel-methods와 완전히 독립적인 새 SD unlearning 프레임워크.

- **부모 프로젝트 코드 import 금지**: `fcf/`, `models/novel/` 코드 참조 없음
- **FCFTrainer 상속 없음**: `LSSETrainer`는 처음부터 독자 설계
- **2-stage 패러다임 폐기**: 단일 최적화 루프, noise 프롬프트 불필요

---

## 구현된 구성 요소

| ID | 이름 | 파일 | FCF 대비 차별점 |
|---|---|---|---|
| **N7** | Concept Null-Space Projection | `methods/cnp.py` | noise 불필요 — SVD로 개념 방향 추출 후 해당 성분 최소화 |
| **N8** | Contrastive Semantic Retention | `methods/csr.py` | MSE retain → InfoNCE. retain/forget 공간 분리 강제 |
| **N9** | CAP-Guided Layer Masking | `methods/clm.py` | N1 CAP 결과 → top-K 레이어만 학습 |

---

## 핵심 하이퍼파라미터

| 파라미터 | 기본값 | 의미 |
|---|---|---|
| `learning_rate` | 2.5e-5 | FCF와 동일 (공정 비교) |
| `alpha` | 1.0 | CNP forget 손실 가중치 |
| `beta` | 1.0 | CSR retain 손실 가중치 |
| `gamma` | 0.5 | implicit CNP 손실 가중치 |
| `temperature` | 0.07 | CSR InfoNCE 온도 τ |
| `clm_top_k` | 3 | 학습 레이어 수 (L0, L1, L2) |
| `seed` | 42 | 재현성 |

---

## 자주 쓰는 커맨드

> PowerShell 환경 — 한 줄 작성, `\` 줄 연속 금지

```powershell
# 기본 학습 (N7+N8, uniform CLM)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train_lsse.py --config configs\nudity_lsse.yaml

# N9 CAP 연동 (기존 CAP JSON 활용)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train_lsse.py --config configs\nudity_lsse.yaml --cap_file ..\fcf-novel-methods\outputs\cap\nudity_heatmap.json

# N8 확장 모드 (forget을 CSR negative로)
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" train_lsse.py --config configs\nudity_lsse.yaml --use_extended_csr

# 단위 테스트
& "c:\Users\vip\anaconda3\envs\fcf\python.exe" -m pytest tests\ -v

# ASR 평가 (로컬 evaluate.py 사용)
python evaluate.py --encoder_dir outputs\lsse_nudity\final --concept nudity --eval_type asr --output_dir outputs\eval\lsse_asr
```

---

## N1/N5/N6 통합 가능성 평가

| 기존 방법 | 통합 여부 | 방법 |
|---|---|---|
| **N1 CAP** | ✅ 즉시 가능 | `--cap_file`로 CLM에 연결 — 구현 완료 |
| **N5 Spherical** | ✅ 가능 | `compute_concept_direction`를 spherical PCA로 교체 (N10) |
| **N6 OT Noise** | ⚠️ 부분적 | LSSE는 noise 불필요. OT 점수로 explicit 프롬프트 가중치 부여 가능 (N13) |

---

## 아이디어 브레인스토밍 — Novel Ideas 백로그

### 즉시 실험 가능

**N10 — Manifold-Aware Concept Direction (MACD)**
- CLIP 임베딩이 narrow cone에 분포 → Euclidean PCA 대신 spherical PCA
- 구현: `compute_concept_direction`에서 L2 정규화 후 Fréchet mean 반복
- 예상 효과: c_dir 정확도 향상 → CNP 손실 집중력 향상

**N11 — Progressive Layer Unlocking (PLU)**
- 현재 CLM은 고정 top-K. 학습 초반 1개 → 후반 K개로 점진 개방
- 구현: epoch 비율에 따라 `apply_uniform_mask(top_k=...)` 동적 변경
- 예상 효과: 초기 급격한 gradient의 downstream 레이어 오염 완화

**N12 — Dual-Direction Forgetting (DDF)**
- 현재: projection 스칼라 → 0 (최소화)
- 강화: `target = z - proj(z, c_dir)` 를 MSE target으로 직접 학습
- 예상 효과: 더 빠른 수렴 + 명확한 기하학적 목표

### 중기 연구 아이디어

**N13 — Concept Vocabulary Contrastive (CVC)**
- N6 OT 점수 높은 noise를 explicit의 hard negative로 CSR에 추가
- N6와 N8의 시너지 활용

**N14 — Adaptive Loss Weighting (ALW)**
- α, β, γ를 고정값 대신 학습 상태에 따라 동적 조정
- 초기: α 높게 (적극적 망각) → 후기: β 높게 (retain 보호)
- GradNorm 또는 uncertainty weighting 기법 적용

**N15 — Cross-Attention Targeted Erasure (CATE)**
- CLIP self-attention head 수준 분석
- 개념 토큰에 높은 attention을 주는 head만 선택 학습
- Layer보다 더 세밀한 granularity

### 장기 연구 방향

**N16 — ROME-style Rank-1 Update (N1 완성)**
- N1 CAP 최고 인과 레이어 → ROME 스타일 weight 직접 편집
- 학습 루프 없이 단일 forward/backward
- Meng et al. ROME (NeurIPS 2022) 참조

**N17 — Multi-Concept Joint Erasure**
- nudity + violence 동시 망각
- concept_dirs 배열 관리 + 방향 간 직교성 보장

---

## 코드 수정 체크리스트

1. 부모 `fcf/`, `train_fcf.py`, `evaluate.py`를 수정하지 않는가?
2. `models/novel/` 코드를 import하지 않는가?
3. 결과를 `lsse/outputs/`에만 저장하는가?
4. `pytest lsse/tests/ -v` 전부 통과하는가?
5. `alpha`, `beta`, `gamma` 변경 시 `configs/nudity_lsse.yaml` 동기화했는가?
