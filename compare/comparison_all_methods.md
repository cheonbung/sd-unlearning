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

## 표 A — 통합 표 (우리 harness, mean ASR 오름차순 = 안전한 순)

**★ = 이번에 재현한 레퍼런스 비교모델**(ESD 학습 · SLD 무학습 재구현 · Safe-CLIP 공개가중치), 전부 우리
harness 동일 프로토콜로 측정.

| 순위 | 모델 | 계열 | 개입 지점 | base | ASR↓ | COCO-FID↓ | COCO-CLIP↑ | LPIPS↓ | IQ↑ |
|---|---|---|---|---|---|---|---|---|---|
| 🥇 1 | **ODACE v3** | ODACE | UNET cross-attn (full) | v1.4 | **4.0** | 118.89 | 25.27 | 0.430 | 0.507 |
| 🥇 1 | **ODACE v1.5** | ODACE | UNET cross-attn (full) | v1.5 | **4.0** | 117.90 | 25.23 | 0.464 | 0.507 |
| 3 | safe_neg | 추론시 | nudity 네거티브 프롬프트 | v1.5 | 15.2 | 120.98 | 25.46 | 0.405 | 0.508 |
| 4 | Sph+OT (N5+N6) | FCF-novel | 텍스트인코더 | v1.4 | 15.6 | 121.41 | 23.92 | 0.461 | 0.508 |
| 5 | LSSE +PLU+W2 | LSSE | 텍스트인코더 | v1.4 | 20.8 | 144.59 | 19.19 | 0.609 | 0.507 |
| 6 | **★ ESD-u** (재현) | ESD | UNET 비-cross-attn | v1.4 | **21.6** | 116.89 | 25.38 | 0.405 | 0.508 |
| 6 | LSSE +PLU | LSSE | 텍스트인코더 | v1.4 | 21.6 | 142.88 | 19.46 | 0.605 | 0.507 |
| 8 | **★ Safe-CLIP** (재현) | Safe-CLIP | CLIP 텍스트인코더 교체 | v1.4 | 44.0 | 119.06 | 25.70 | 0.387 | 0.508 |
| 9 | **★ SLD-Max** (재현) | SLD | 추론 가이던스(δ0,sS5000) | v1.4 | 45.2 | 121.65 | 24.41 | 0.477 | 0.509 |
| 10 | vanilla LSSE (N7+N8+N9) | LSSE | 텍스트인코더 | v1.4 | 46.0 | 147.71 | 24.20 | 0.641 | 0.506 |
| 11 | DACE v2 (concept-axis) | DACE | 텍스트인코더 | v1.4 | 50.8 | 121.29 | 25.78 | 0.406 | 0.507 |
| 12 | FCF-P (원조 논문) | FCF | 텍스트인코더 | v1.4 | 52.8 | 120.22 | 25.30 | 0.391 | 0.508 |
| 13 | **SD2.1-base** (NSFW-필터 사전학습) | 기준선 | (소거 없음) | v2.1 | 54.4 | 118.32 | 25.95 | — | 0.508 |
| 14 | ODACE v2 (K/V-only, 약한 편집) | ODACE | UNET K/V | v1.4 | 56.0 | 116.39 | 25.98 | 0.265 | 0.508 |
| 15 | **★ SLD-Strong** (재현) | SLD | 추론 가이던스(δ7,sS2000) | v1.4 | 58.4 | 121.58 | 24.92 | 0.349 | 0.508 |
| 16 | raw v1.5 | 기준 | — | v1.5 | 60.0 | 117.99 | 26.45 | 0 (기준) | 0.508 |
| 17 | FCF-E (원조 논문) | FCF | 텍스트인코더 | v1.4 | 61.2 | 119.33 | 25.58 | 0.345 | 0.508 |
| 18 | raw v1.4 | 기준 | — | v1.4 | 62.0 | 118.64 | 26.48 | 0 (기준) | 0.508 |
| 18 | **★ SLD-Medium** (재현) | SLD | 추론 가이던스(δ10,sS1000) | v1.4 | 62.0 | 118.98 | 25.78 | 0.237 | 0.508 |
| 20 | DACE+PLU (concept-axis+PLU) | DACE | 텍스트인코더 | v1.4 | 73.6 | 123.71 | 25.82 | 0.387 | 0.507 |

*(이번 실행으로 **전 모델 표준 COCO 측정 완료**(빈칸 "—" 제거). **LPIPS↓** = 동일 base raw와 같은 캡션·시드
생성쌍의 지각거리(편집 드리프트; 낮을수록 일반 생성을 raw 근처로 보존). **IQ↑** = CLIP-IQA "Good/Bad photo"
확률(무참조 화질). raw 두 행은 자기 자신 기준이라 LPIPS=0; SD2.1-base는 동일 base raw가 없어 "—".)*

**LPIPS·IQ 해석:**
- **LSSE 계열(0.61–0.64)이 압도적으로 높다** = 일반 생성이 raw에서 크게 드리프트(FID 143–148·CLIP 19로 동반
  악화) → 텍스트인코더 과편집이 locality를 망친다. 효능(ASR 20.8)이 좋아도 품질 대가가 가장 크다.
- **ODACE는 효능-locality를 동시 달성:** ASR 4.0인데 LPIPS도 중간대(0.43–0.46)·FID raw급(117.9–118.9) →
  nudity만 제거하고 일반 생성은 raw 근처 보존. ESD-u(0.405)·Safe-CLIP(0.387)·FCF-P(0.391)도 보존은 양호.
- **단, LPIPS는 단독으로 읽으면 오해:** ODACE v2(0.265)·SLD-Medium(0.237)처럼 *약·얕은* 개입은 LPIPS가
  낮아도 ASR이 나쁘다(56·62). 아무것도 안 바꾸면 LPIPS≈0이므로 **반드시 ASR과 함께** 읽어야 한다.
- **IQ(CLIP-IQA)는 0.5055–0.5086으로 전 모델 거의 평탄** → 본 COCO 셋에서 화질 변별력이 낮다(모두
  비슷하게 "그럴듯한" 이미지를 생성). locality 변별은 FID/LPIPS가 담당하고 IQ는 보조 지표로 둔다.

### 재현 비교모델 per-attack ASR (우리 harness, ×50장)

| 모델 | I2P | Ring-A-Bell | RaB(Re) | P4D | UnlearnDiffAtk | mean |
|---|---|---|---|---|---|---|
| ★ ESD-u | 22.0 | 32.0 | 14.0 | 8.0 | 32.0 | **21.6** |
| ★ Safe-CLIP | 24.0 | 68.0 | 52.0 | 36.0 | 40.0 | 44.0 |
| ★ SLD-Max | 22.0 | 76.0 | 56.0 | 18.0 | 54.0 | 45.2 |
| ★ SLD-Strong | 34.0 | 94.0 | 76.0 | 26.0 | 62.0 | 58.4 |
| ★ SLD-Medium | 42.0 | 94.0 | 82.0 | 26.0 | 66.0 | 62.0 |
| raw v1.4 (참고) | 42.0 | 86.0 | 84.0 | 32.0 | 66.0 | 62.0 |
| ODACE v3 (참고) | 8.0 | 0.0 | 0.0 | 2.0 | 10.0 | **4.0** |

- **Ring-A-Bell이 모든 재현 레퍼런스를 무너뜨린다:** SLD-Medium/Strong은 RaB 94(raw 86보다 *악화*),
  Safe-CLIP 68, ESD-u 32. ODACE만 0. 적대적 최적화 프롬프트가 추론·텍스트인코더 개입을 우회함을 직접 확인.
- **ESD-u가 재현 레퍼런스 중 최강**(21.6) — UNET 가중치 편집이라 추론·텍스트 개입보다 강건. 그래도
  ODACE(4.0)의 1/5 수준. 같은 UNET 개입이라도 *출력-접지 cross-attn 편집*(ODACE)이 캐논 ESD보다 깊다.
- **SLD 강도 단조성:** Medium 62.0 → Strong 58.4 → Max 45.2. 강하게 걸수록 안전하지만 Max도 RaB 76.

## 텍스트인코더 계열의 과거 품질값 (다른 프로토콜 — 본 표와 직접 비교 불가)

| 모델 | CLIP | FID | 프로토콜 |
|---|---|---|---|
| raw SD (기준) | 26.45 | — | fcf harness |
| FCF-P | 26.43 | 309.8 | COCO-2k (fcf harness) |
| Sph+OT (N5+N6) | 26.01 | 316.1 | COCO-2k (fcf harness) |
| ODACE v3 (self-cal) | 26.79 (raw 27.11, Δ−0.32) | 드리프트 178.4 ≤ 노이즈 floor 184.5 (excess −6.1) | retain 85프롬프트 자기보정 |

- FID 스케일이 3종(COCO-300 ~118 · COCO-2k ~310 · self-cal-85 ~180)이라 절대값 교차 비교 금지.
- 같은 스케일 안에서만: Sph+OT는 최저 ASR(15.6)을 retain CLIP 약간 희생(26.45→26.01)으로 달성.

## 표 B — 원논문 보고값 (as-reported, Phase 0 PDF 추출 초안)

> ⚠️ **표 A와 절대 병합 금지.** 각 논문은 base(SD v1.4)·공격셋·NudeNet 버전/임계·FID refset·CLIP
> 스케일이 서로 다르다. 아래는 `fcf/reference/`의 원 PDF에서 추출한 **저자 보고값**이며, 프로토콜을
> 행마다 명시한다. 우리 harness 재측정값(표 A)과의 차이 자체가 분석 대상(재현 충실도 축).

### B-0. FCF Table 1 — 교차방법 ASR (가장 표 A에 근접한 단일 공격수트)

FCF(Fan 2026)는 ESD·SLD·Safe-CLIP을 **하나의 일관된 공격수트**(I2P 원본 + Ring-A-Bell + RaB(Re) +
P4D + UnlearnDiffAtk)로 함께 측정 → 우리 5-공격 harness와 가장 비교 가능한 외부 표. **Nudity ASR(%) ↓**:

| 공격 | SD v1.4 | ESD | SLD-Med | Safe-CLIP | RECE | Receler | FCF-E | FCF-P |
|---|---|---|---|---|---|---|---|---|
| I2P 원본 | 60.52 | 6.43 | 30.04 | 5.15 | 4.72 | 3.86 | 6.44 | 3.00 |
| Ring-A-Bell | 91.58 | 32.63 | 92.63 | 44.21 | 3.16 | 2.11 | 13.68 | 1.05 |
| Ring-A-Bell(Re) | 53.85 | 3.85 | 38.46 | 11.58 | 4.81 | 2.89 | 1.92 | 0.96 |
| P4D | 81.05 | 50.53 | 66.31 | 48.43 | 28.42 | 26.31 | 7.36 | 5.26 |
| UnlearnDiffAtk | 91.95 | 65.52 | 72.41 | 63.23 | 22.99 | 20.69 | 14.94 | 6.90 |
| **mean(계산)** | **75.79** | **31.79** | **59.97** | **34.52** | 12.82 | 11.17 | 8.87 | **3.43** |

*(mean 행은 5-공격 단순평균 = 본인 계산. 저자는 per-attack만 보고하고 "FCF-P가 baseline 대비 평균
8.91%p 개선"이라 서술. 프로토콜: SD v1.4 / SLD=Default Medium / NudeNet(EXPOSED_ANUS·BREAST_F·
GENITALIA_F·GENITALIA_M) / 고정시드 1장·원프롬프트 기준. **NudeNet 임계·장수·프롬프트셋이 우리
harness(score>0.3, ×50장)와 달라** 표 A의 같은 모델값과 직접 비교 불가.)*

**관전 포인트(재현 시 기대치):** 우리 harness가 FCF-P를 재측정했을 때 **52.8**(표 A)인데 FCF 원논문은
**3.43**(위) — 약 15배 차이. 즉 우리 harness는 FCF 원프로토콜보다 훨씬 적대적이다. 따라서 ESD/SLD/
Safe-CLIP을 우리 harness로 재현하면 표 A의 ASR은 위 표 B 값보다 **상당히 높게** 나올 것으로 예상.

### B-1. 효능(nudity) — 각 논문 자체 프로토콜

| 방법 | 보고 지표 | 원값 | 프로토콜 |
|---|---|---|---|
| **ESD-u** (Gandikota'23, ICCV) | I2P 노출신체 이미지 수 | SD v1.4 **796 → 134** (η=1) | I2P 4703프롬프트, NudeNet 노출분류 카운트. (SD2.0 필터=417) |
| ESD η 절제 | nudity 소거율 / 객체정확도 영향 | η1: 83%/−7% · η3: 88%/−14% · η10: 92%/−34% | 1000-way 분류, 10 Imagenette 클래스 |
| **SLD** (Schramowski'23, CVPR) | I2P inappropriate 확률(Q16∪NudeNet) | SD **.39** → W .29 · M .19 · S .13 · **Max .09** | I2P 4703×10장, Q16+NudeNet 결합(노출생식기만 NudeNet). Sexual: .35→Max .06 |
| **Safe-CLIP** (Poppi'25, ECCV) | I2P Avg inappropriate 확률(%) | SD v1.4 **35.7 → 22.2** (−13.5p) | I2P 4703+ViSU ×5장, Q16+NudeNet. SD v1.4 텍스트인코더 교체. Sexual 24.8→15.9. ViSU Avg 26.2→3.6(−22.6p) |

### B-2. 품질(localitiy) — 각 논문 COCO 프로토콜 (FID/CLIP 스케일 상이)

| 방법 | FID | CLIP | 프로토콜 |
|---|---|---|---|
| ESD Table 1 (SD / ESD-u / ESD-u-3) | 14.50 / **13.68** / 17.27 | 0.1592 / **0.1585** / 0.1586 | COCO-30k, α=7.5, CLIP=raw cosine(~0.15). REAL CLIP 0.1561 |
| SLD Table 2 (SD/W/M/S/Max) | 14.43 / 15.81 / 16.90 / 18.28 / 18.76 | 0.75→0.76 (CLIP **distance**↓) | COCO FID-30k. CLIP distance(낮을수록 좋음) |
| FCF Table 3 (SD/ESD/SLD/Safe-CLIP/RECE/Receler/FCF-E/FCF-P) | 14.51 / 14.95 / 15.53 / 15.49 / 15.08 / 14.89 / 15.01 / 15.07 | 31.35 / 30.13 / 30.85 / 30.48 / 30.64 / 31.02 / 30.89 / 31.03 | COCO-30k(nudity 제외), CLIP=×100 스케일 |

*(세 CLIP 스케일이 다름: ESD raw-cosine ~0.16 · SLD distance ~0.75 · FCF ×100 ~31. FID도 30k(~14–18)
대 우리 COCO-300(~118)로 다름 → 표 A의 COCO 컬럼과 직접 비교 불가.)*

### B-3. 재현 사양 (Phase 1–4 확정)

| 방법 | 개입지점 | 학습? | 핵심 하이퍼파라미터(원논문) | 가용성 |
|---|---|---|---|---|
| **ESD-u** | UNET 비-cross-attn | **예(GPU)** | η=1, lr 1e-5, 1000 step, bs1, Adam, SD v1.4. (nudity=ESD-u, style=ESD-x) | 캐논 recipe 재구현 |
| **SLD** | 추론시 가이던스 | 아니오 | Weak{δ15,sS200,λ0,sm0} · Med{δ10,sS1000,λ.01,sm.3,βm.4} · Strong{δ7,sS2000,λ.025,sm.5,βm.7} · Max{δ0,sS5000,λ1,sm.5,βm.7}. safety concept 20개 텍스트 | 수동 재구현(eqs.3–8 보유). diffusers 0.38 SLD 파이프=dummy stub |
| **Safe-CLIP** | CLIP 텍스트인코더 | 아니오 | ViT-L/14, text dim **768**(SD v1.4 호환) | **`aimagelab/safeclip_vit-l_14`** (HF 가용, 5파일, CLIPModel→`.text_model` 사용) |

## 다방면 분석 (Phase 6)

### ③ 재현 충실도 — 우리 harness(표 A) vs 원논문/외부 보고값(표 B)

| 방법 | 우리 harness ASR | 비교 기준값 | 해석 |
|---|---|---|---|
| **SLD-Medium** | 62.0 | FCF 보고 SLD-Med nudity mean **60.0** | Δ+2.0 — **거의 정확히 재현**. SLD가 적대공격에 약하다는 FCF 관찰 확인 |
| **Safe-CLIP** | 44.0 | FCF 보고 Safe-CLIP mean **34.5** | 같은 차수(Δ+9.5). 우리 harness(고정 공격셋 ×50)가 FCF보다 적대적 |
| **ESD-u** | 21.6 | FCF 보고 ESD mean 31.8 / 원논문 I2P 796→134(−83%) | 방향 일치(raw 62.0→21.6, −65%). ESD가 "중간 강도" 소거라는 점 재현 |

- **핵심:** 우리 harness는 원프로토콜보다 **체계적으로 더 적대적**이다(FCF-P가 원논문 3.4 → 우리 52.8로
  치솟은 것과 동일 방향). 그럼에도 SLD-Medium은 60↔62로 거의 일치 → **재현 충실도 신뢰 가능**.
- **표 A/B 병합 절대 금지**가 정당함을 데이터가 직접 보여줌(같은 방법이 프로토콜에 따라 크게 다른 값).

### ④ 개입 지점별 강건성 (mean ASR↓, 우리 harness)

| 개입 지점 | 방법(ASR) | 패턴 |
|---|---|---|
| 추론시 가이던스 | safe_neg 15.2 · SLD-Max 45.2 · SLD-Strong 58.4 · SLD-Med 62.0 | 가중치 미변경 → 적대공격에 가장 취약(RaB 76~94) |
| CLIP 텍스트인코더 | Safe-CLIP 44.0 · Sph+OT 15.6 · FCF-P 52.8 · DACE 50.8~73.6 | 텍스트임베딩 proxy는 ASR을 underdetermine(분산 큼) |
| 사전학습 NSFW 필터 | SD2.1-base 54.4 | 잠재표현 잔존 → 적대 복원 취약 |
| **UNET 비-cross-attn** | **ESD-u 21.6** | 가중치 직접 편집 → 추론·텍스트보다 강건 |
| **UNET cross-attn(출력접지)** | **ODACE 4.0** | 출력 노이즈 직접 최적화 → 최심부 소거 |

- 개입이 **출력(UNET)에 가까울수록** 강건: 추론(45~62) < 텍스트인코더(15~73, 분산↑) < UNET(ESD-u 21.6,
  ODACE 4.0). 같은 UNET이라도 ODACE의 **출력-접지 cross-attn**이 ESD의 음성가이던스 비-cross-attn보다 5배 깊다.
- safe_neg(15.2)가 SLD(45~62)보다 훨씬 낮은 건, safe_neg는 nudity 전용 네거티브인 반면 SLD는 20개 개념
  광역 안전가이던스라 nudity 집중도가 낮기 때문(트레이드오프 차이).

### ⑤ 연산 비용 / 개입 규모

| 방법 | 학습 | 비용(RTX 4070) | 편집 파라미터 |
|---|---|---|---|
| SLD (Med/Strong/Max) | **무학습** | 0 (추론시 3-way 예측, ~1.5× 추론) | 0 |
| Safe-CLIP | **무학습** | 0 (공개 가중치 텍스트인코더 교체) | 0(사전학습된 것 사용) |
| ESD-u | 학습 | ~50분 / 1000 step | **815.6M**(UNET의 95%, 비-cross-attn 전체) |
| ODACE v3 | 학습 | ~유사 / 1500 step | cross-attn(소수) |

- ESD-u는 UNET의 95%를 편집(가장 침습적)하고도 ODACE(소수 cross-attn 편집)보다 ASR이 5배 높다 →
  **"얼마나 많이 건드리나"가 아니라 "무엇을 목적함수로 누르나"가 결정적**(출력접지 vs 음성가이던스).

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

6. **재현한 캐논 레퍼런스(ESD·SLD·Safe-CLIP)를 ODACE가 전부 지배.** 동일 harness에서 ESD-u **21.6**
   (재현 레퍼런스 최강) · Safe-CLIP **44.0** · SLD-Max **45.2**~SLD-Med **62.0** 모두 ODACE(4.0)·
   safe_neg(15.2)보다 위. SLD-Medium은 우리 재현 62.0 ↔ FCF 보고 60.0으로 **충실히 재현**됐고, Ring-A-Bell
   에서 94(raw 86보다 악화)까지 뚫린다 → 추론·텍스트인코더·사전학습필터 개입은 적대공격에 구조적으로
   취약. ESD-u가 UNET의 95%를 학습편집하고도 ODACE의 출력-접지 cross-attn 편집(소수 파라미터)에 5배
   뒤진다는 점이, **개입 규모가 아니라 출력-접지 목적함수가 깊은 소거의 핵심**임을 재확인한다.

## 출처
- ASR: `lsse/outputs/comparison_unified.md` (텍스트인코더 계열) · `xmodel/outputs/<label>/metrics.json` (교차모델)
- COCO-FID/CLIP: `xmodel/outputs/<label>/coco_metrics.json`
- 과거 품질값: `lsse/outputs/comparison_unified.md` (fcf harness 저장값) · `odace/evaluate_utility.py` (self-cal)
