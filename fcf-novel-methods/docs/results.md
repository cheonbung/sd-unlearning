# FCF 실험 결과 (측정치 기록)

마지막 업데이트: 2026-04-22

---

## Table 1 — ASR (Attack Success Rate)

### Nudity (NudeNet v3, 공정 비교 완료)

| 공격 유형 | Baseline | FCF-P | FCF-E |
|---|---|---|---|
| I2P | 36.00% | 18.00% | 28.00% |
| Ring-A-Bell | 92.00% | 84.00% | 94.00% ↑ |
| Ring-A-Bell(Re) | 82.74% (4750장) | 93.94% (4750장) ↑ | 90.08% (4750장) ↑ |
| P4D | 32.00% | 16.00% | 28.00% |
| UnlearnDiffAtk | 60.00% | 48.00% | 62.00% ↑ |

Ring-A-Bell(Re): 각 모델을 타겟으로 최적화된 적응형 공격 → FCF-P/E ASR이 Baseline보다 높은 것은 예상된 결과.

### Violence (Q16 classifier, 공정 비교 완료)

| 공격 유형 | Baseline | FCF-P | FCF-E |
|---|---|---|---|
| I2P | 43.06% (757장) | **26.82%** (757장) | 36.20% (757장) |
| Ring-A-Bell | 89.59% (269장) | **20.07%** (269장) | 72.49% (269장) |
| UnlearnDiffAtk | 43.06% (757장) | **26.82%** (757장) | 36.20% (757장) |

참고: I2P와 UnlearnDiffAtk 파일은 내용이 동일(마지막 개행만 다름) → ASR 수치 같은 것이 정상.
Violence P4D / Ring-A-Bell(Re): 데이터 파일 없음 — 미측정.

---

## Table 3 — FID + CLIP Score (Generative Quality, COCO-2017 val 기준)

### Nudity

| 모델 | CLIP Score ↑ | FID ↓ |
|---|---|---|
| Baseline | 26.45 | 315.29 |
| FCF-P | 26.39 | 303.24 |
| FCF-E | 26.32 | 313.50 |

### Violence

| 모델 | CLIP Score ↑ | FID ↓ |
|---|---|---|
| Baseline | 26.57 | 300.30 |
| FCF-P | 26.58 | 285.34 |
| FCF-E | 26.46 | 308.98 |

### Van Gogh

| 모델 | CLIP Score ↑ | FID ↓ |
|---|---|---|
| Baseline | 28.45 | 333.19 |
| FCF-P | 27.50 | 316.42 |
| FCF-E | 27.85 | 326.29 |

---

## Table 2 — LPIPS (Van Gogh 스타일 망각, 20장 기준)

| 모델 | LPIPS_f ↑ | LPIPS_m ↓ | LPIPS_d ↑ |
|---|---|---|---|
| FCF-P | 0.7004 | 0.3846 | 0.3157 |
| FCF-E | 0.7164 | 0.2649 | 0.4515 |

- LPIPS_f: 스타일 망각량 (높을수록 잘 잊음)
- LPIPS_m: 비타겟 이미지 변화량 (낮을수록 보존 잘됨)
- LPIPS_d: 판별력 = LPIPS_f − LPIPS_m

---

## fcf-novel-methods (2026-05-22 추가)

CLIP 텍스트 인코더에 N5 manifold variants + N6 OT noise vocabulary 적용. Nudity 개념 대상.

### ASR (Nudity, NudeNet v3, 50장)

| 공격 | N5 Spherical | N5 Euclidean | **Euclidean+OT (N6)** | **Spherical+OT (N5+N6)** |
|---|---|---|---|---|
| I2P | 22.0% | 14.0% | 14.0% | **14.0%** |
| Ring-A-Bell | 40.0% | 70.0% | 34.0% | **24.0%** |
| Ring-A-Bell(Re) | 96.0% | 98.0% | 28.0% | **20.0%** |
| P4D | 20.0% | 12.0% | 12.0% | **10.0%** |
| UnlearnDiffAtk | 38.0% | 44.0% | 18.0% | **14.0%** |

핵심 관찰: N6 OT noise vocabulary 적용 시 Ring-A-Bell(Re) ASR이 극적으로 감소 (96→20%, 98→28%). 적응형 공격에 대한 강건성 확보.

### Quality (CLIP Score + FID, COCO-2017 val ref, 30장)

| 메트릭 | Spherical | Euclidean | Euclidean+OT | Spherical+OT |
|---|---|---|---|---|
| CLIP Score ↑ | 26.12 | **26.43** | 26.17 | 26.01 |
| FID ↓ | **305.35** | 309.83 | 314.71 | 316.09 |

### N1 CAP (Causal Activation Patching) — 버그 수정 후

- 버그 원인: 레이어 i 출력 전체를 noise activation으로 교체 → 후속 layer가 결정론적으로 같은 endpoint로 수렴 → 모든 레이어 동일 점수 (0.7884)
- 수정: 단일 토큰 위치(1)만 패치 → attention mixing이 레이어별로 차별화

Nudity 개념의 layer-wise causal scores (CLIP-L/14, 12 layers):

| Layer | Score | Layer | Score |
|---|---|---|---|
| 0 | **0.4952** | 6 | 0.3323 |
| 1 | 0.4110 | 7 | 0.3281 |
| 2 | 0.3721 | 8 | 0.3293 |
| 3 | 0.3491 | 9 | 0.2561 |
| 4 | 0.3390 | 10 | 0.1870 |
| 5 | 0.3374 | 11 | 0.0529 |

해석: 초기 레이어 (0–3)가 nudity 개념 인코딩에 가장 큰 인과적 기여. 후기 레이어는 영향 미미.

## 미완료 항목

- Violence P4D / Ring-A-Bell(Re): 프롬프트 데이터 파일 없음
- 비교 베이스라인 (ESD, CA, SLD): 레퍼런스 논문 제공 시 진행 예정
- CAP을 활용한 layer-targeted FCF training (top-3 layers만 unfreeze): 후속 연구
