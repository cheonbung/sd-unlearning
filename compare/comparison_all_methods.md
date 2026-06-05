# 전체 방법 통합 비교 — FCF · LSSE · DACE · ODACE · 안전 기준선

이 프로젝트에서 실험한 **모든 모델**을 하나의 표로 정리한다. 두 기존 문서를 통합한다:
- `lsse/outputs/comparison_unified.md` — 텍스트인코더 계열(FCF/LSSE/DACE) + ODACE의 단일-harness ASR
- `compare/comparison_models.md` — 교차모델(SD v1.4/v1.5/v2.1 + 안전 기준선)의 ASR + 표준 COCO

## ⚠️ 측정 프로토콜 (비교 가능성의 핵심)

- **ASR↓ (효능): 전 모델 동일 프로토콜 → 직접 비교 가능.** NudeNet v3(score>0.3), 5공격(I2P,
  Ring-A-Bell, RaB(Re), P4D, UnlearnDiffAtk) × 50장, guidance 7.5 · 50 steps. 텍스트인코더 계열은
  lsse harness(SD-v1-4 디코딩)로, 교차모델 계열은 `xeval.py`(각 모델 자체 base)로 측정 — ASR은
  "공격 이미지의 nudity 비율"이라 base와 무관하게 안전도로 비교된다.
- **품질(FID/CLIP): 프로토콜이 달라 한 컬럼으로 섞을 수 없음.** 아래 표의 COCO-FID/CLIP은
  `eval_coco.py`(COCO val 캡션 300장 생성 vs 실제 600장, FID 스케일 ~118)로 측정한 **교차모델 6종만**
  값이 있다. FCF/LSSE/DACE 계열의 과거 품질값은 *다른 스케일*이라 본 표에 넣지 않고 아래 별도 절에 둔다.

## 통합 표 (mean ASR 오름차순 = 안전한 순)

| 순위 | 모델 | 계열 | 개입 지점 | base | ASR↓ | COCO-FID↓ | COCO-CLIP↑ |
|---|---|---|---|---|---|---|---|
| 🥇 1 | **ODACE v3** | ODACE | UNET cross-attn (full) | v1.4 | **4.0** | 118.89 | 25.27 |
| 🥇 1 | **ODACE v1.5** | ODACE | UNET cross-attn (full) | v1.5 | **4.0** | 117.90 | 25.23 |
| 3 | safe_neg | 추론시 | nudity 네거티브 프롬프트 | v1.5 | 15.2 | 120.98 | 25.46 |
| 4 | Sph+OT (N5+N6) | FCF-novel | 텍스트인코더 | v1.4 | 15.6 | — | — |
| 5 | LSSE +PLU+W2 | LSSE | 텍스트인코더 | v1.4 | 20.8 | — | — |
| 6 | LSSE +PLU | LSSE | 텍스트인코더 | v1.4 | 21.6 | — | — |
| 7 | vanilla LSSE (N7+N8+N9) | LSSE | 텍스트인코더 | v1.4 | 46.0 | — | — |
| 8 | DACE v2 (concept-axis) | DACE | 텍스트인코더 | v1.4 | 50.8 | — | — |
| 9 | FCF-P (원조 논문) | FCF | 텍스트인코더 | v1.4 | 52.8 | — | — |
| 10 | **SD2.1-base** (NSFW-필터 사전학습) | 기준선 | (소거 없음) | v2.1 | 54.4 | 118.32 | 25.95 |
| 11 | ODACE v2 (K/V-only, 약한 편집) | ODACE | UNET K/V | v1.4 | 56.0 | — | — |
| 12 | raw v1.5 | 기준 | — | v1.5 | 60.0 | 117.99 | 26.45 |
| 13 | FCF-E (원조 논문) | FCF | 텍스트인코더 | v1.4 | 61.2 | — | — |
| 14 | raw v1.4 | 기준 | — | v1.4 | 62.0 | 118.64 | 26.48 |
| 15 | DACE+PLU (concept-axis+PLU) | DACE | 텍스트인코더 | v1.4 | 73.6 | — | — |

*("—" = 표준 COCO 프로토콜 미측정. 해당 모델들의 과거 품질값은 아래 별도 절 참조.)*

## 텍스트인코더 계열의 과거 품질값 (다른 프로토콜 — 본 표와 직접 비교 불가)

| 모델 | CLIP | FID | 프로토콜 |
|---|---|---|---|
| raw SD (기준) | 26.45 | — | fcf harness |
| FCF-P | 26.43 | 309.8 | COCO-2k (fcf harness) |
| Sph+OT (N5+N6) | 26.01 | 316.1 | COCO-2k (fcf harness) |
| ODACE v3 (self-cal) | 26.79 (raw 27.11, Δ−0.32) | 드리프트 178.4 ≤ 노이즈 floor 184.5 (excess −6.1) | retain 85프롬프트 자기보정 |

- FID 스케일이 3종(COCO-300 ~118 · COCO-2k ~310 · self-cal-85 ~180)이라 절대값 교차 비교 금지.
- 같은 스케일 안에서만: Sph+OT는 최저 ASR(15.6)을 retain CLIP 약간 희생(26.45→26.01)으로 달성.

## 핵심 결론

1. **ODACE v3/v1.5(ASR 4.0)가 전 방법 최강 효능.** 2위 Sph+OT(15.6)·교차모델 safe_neg(15.2)를
   ~4배 앞선다. 텍스트인코더 계열의 floor(~15–20)는 *개입 지점*의 한계였고, 개입을 **UNET
   cross-attention**으로 옮기자 floor가 깨졌다(DACE 음성 결과 → ODACE 양성 검증).

2. **ODACE의 낮은 ASR은 "파괴적 소거"가 아니다.** 표준 COCO에서 ODACE의 일반 생성은 raw와
   구별 불가(FID 118.9 ≈ 118.6, CLIP −1.2 ≈ safe_neg −1.0). off-target은 *적대 nudity 프롬프트에만*
   국한 = 사실상 nudity 거부.

3. **필터링된 사전학습 ≠ 강건한 소거.** SD2.1-base는 충실도 정상(FID 118.32, CLIP 25.95)이지만
   적대 공격엔 취약(ASR 54.4, Ring-A-Bell 82) — raw v1.4(62.0)와 큰 차이 없음. ODACE가 같은
   충실도에서 ASR 4.0으로 **상한선 기준점을 압도**.

4. **DACE는 결정적 음성 결과.** concept_shift(텍스트임베딩 요약지표)를 직접 누를수록 ASR이
   오히려 악화(DACE+PLU 73.6, raw보다 나쁨) → 텍스트임베딩 proxy는 ASR을 underdetermine.

5. **재현성:** LSSE +PLU 계열은 4시드 std≈0. ODACE는 base 버전 간 전이(v1.5 ≈ v1.4).

## 출처
- ASR: `lsse/outputs/comparison_unified.md` (텍스트인코더 계열) · `xmodel/outputs/<label>/metrics.json` (교차모델)
- COCO-FID/CLIP: `xmodel/outputs/<label>/coco_metrics.json`
- 과거 품질값: `lsse/outputs/comparison_unified.md` (fcf harness 저장값) · `odace/evaluate_utility.py` (self-cal)
