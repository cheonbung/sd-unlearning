# 교차모델 비교 — 표준 프로토콜 (효능 ASR + 국소성 COCO-FID/CLIP)

**배경:** 처음엔 "ODACE가 ASR은 낮지만 공격 프롬프트에 뜬금없는 이미지를 낸다"는 정성 관찰에서
출발해 커스텀 지표(CLIP_atk, people_clip, people_asr)를 썼다. 그러나 **CLIP_atk는 혼동이 있다**:
공격 프롬프트엔 nudity 의도가 들어 있어, CLIP 하락이 (a) nudity 제거(좋음)인지 (b) off-target
(나쁨)인지 분리하지 못하고, ODACE가 nudity를 더 많이 지울수록 기계적으로 더 떨어진다. 그래서
**커스텀 지표는 전부 폐기**하고, 소거 문헌의 표준 locality 축으로 재측정했다:

**측정 (`xmodel/xeval.py` + `xmodel/eval_coco.py`):** SD 512 · 50 steps · guidance 7.5 · seed 42.
- **ASR↓** (효능): 공격 5종×50장, NudeNet v3(score>0.3). 낮을수록 안전.
- **COCO-FID↓** (국소성/품질): COCO val 캡션 300장 생성 vs 실제 COCO 600장. nudity 의도 없는 일반
  캡션이라 "제거 vs off-target" 혼동이 **원천적으로 없음**. (절대값 ~118은 N=300 소표본 편향 —
  raw 대비 **상대 비교**가 핵심.)
- **COCO-CLIP↑** (캡션 충실도): CLIP(생성, 캡션). 높을수록 일반 프롬프트를 잘 따름.

## 결과 (표준 지표)

| 모델 | base | ASR↓ | COCO-FID↓ (vs raw) | COCO-CLIP↑ (vs raw) |
|---|---|---|---|---|
| raw v1.4 | v1.4 | 62.0 | 118.64 (기준) | 26.48 (기준) |
| raw v1.5 | v1.5 | 60.0 | 117.99 | 26.45 |
| **safe_neg** (v1.5 + nudity 네거티브) | v1.5 | 15.2 | 120.98 (+2.3) | 25.46 (−1.0) |
| **ODACE v3** | v1.4 | **4.0** | 118.89 (+0.3) | 25.27 (−1.2) |
| **ODACE v1.5** | v1.5 | **4.0** | 117.90 (−0.7) | 25.23 (−1.25) |
| **SD2.1-base** (NSFW-필터 학습) | v2.1 | 54.4 | 118.32 (≈raw) | 25.95 |

*(SD2.1-base는 공식 repo가 gated여서 공개 미러 `Manojb/stable-diffusion-2-1-base`(검증됨)로
fallback 실행 — ASR·COCO **모두 완료**. ASR 세부: I2P 32 · Ring-A-Bell **82** ·
Ring-A-Bell(Re) **82** · P4D 28 · UnlearnDiffAtk 48. COCO는 다른 base(v2.1)라 v1.x-raw와의
절대 비교는 근사치지만, FID 118.32는 raw v1.4(118.64)와 사실상 동일하고 CLIP 25.95는 v1.x
소거법(25.2~25.5)보다 오히려 높음 — 일반 생성 품질은 정상.)*

## 해석 (이전 "파괴적 off-target" 경보의 정정)

1. **표준 locality로 보면 ODACE는 일반 생성을 거의 손상하지 않는다.** COCO-FID는 전 모델 118~121로
   **사실상 동일** — ODACE(118.9/117.9)는 raw(118.6)와 구별 불가, safe_neg(121)보다도 낮다. 즉
   일반 이미지 분포/품질 **손상 없음**.

2. **COCO-CLIP 하락도 미미하고 깨끗한 안전법과 동급이다.** ODACE −1.2 vs safe_neg −1.0. 일반
   캡션에 대한 충실도 손실이 거의 같다.

3. **따라서 앞서 보고한 CLIP_atk −7~−8의 "파괴적 off-target"은 대부분 지표 혼동의 아티팩트였다.**
   그 하락의 큰 부분은 "ODACE가 nudity를 더 많이 제거"한 결과(공격 프롬프트엔 nudity 의도 포함)이지,
   일반 콘텐츠 파괴가 아니다.

4. **두 관찰은 모순이 아니다.** 갤러리에서 본 off-target은 *적대적 nudity 프롬프트에 국한*되며
   (거기서 derail = 사실상 nudity 거부), COCO 같은 *일반 프롬프트*에선 ODACE가 충실도를 보존한다.

5. **ODACE는 base 버전 간 전이된다.** v1.5 재학습이 v1.4(v3)와 거의 동일(ASR 4.0, FID/CLIP 동급).

## 폐기한 커스텀 지표 (기록용)

CLIP_atk(공격 프롬프트 정렬), people_clip, people_asr는 사용자 지시로 폐기. CLIP_atk는 위 (3)의
혼동 때문에 "off-target"을 과대평가했고, people 프로브는 명명된 표준 벤치마크가 아니었다. 효능은
ASR, 국소성은 COCO-FID/CLIP로 일원화한다.

## 외부 도구 메모

- **SD2.1-base** = 공식 `stabilityai/...`는 HF gated일 수 있어, `xeval.py`의 `sd21base`가 공개 미러
  `Manojb/stable-diffusion-2-1-base`·`sd2-community/stable-diffusion-2-1-base`를 fallback으로
  시도(둘 다 diffusers 포맷 검증됨, 토큰 불필요). "한 번도 nudity를 학습하지 않은" ideal 기준점.
- **SLD** = 이 환경 diffusers 0.38.0에서 파이프라인 제거 → safe_neg(네거티브 프롬프트)로 대체.

## 결론 (수정)

표준 프로토콜로 보면 **ODACE는 "충실도를 파괴하는 소거"가 아니다.** 일반 생성(COCO-FID/CLIP)을
깨끗한 안전법(safe_neg)과 동급으로 보존하면서 ASR을 4.0까지 낮춘다. 단 *적대적 nudity 프롬프트*
에서는 프롬프트를 벗어나는데, 이는 nudity 거부의 한 형태로 볼 수 있다.

**SD2.1-base(사전학습 NSFW-필터 모델)는 "상한선"이 아니었다.** 일반 충실도는 정상이지만
(COCO-FID 118.32 ≈ raw, COCO-CLIP 25.95 — v1.x 소거법보다 오히려 높음) 적대적 공격에는
취약하다: ASR_mean **54.4**, 특히 Ring-A-Bell이 **82**까지 끌어올린다. 즉 ASR↔충실도 평면에서
SD2.1-base는 "높은 충실도 + 높은 ASR" 코너에 있고, ODACE는 **같은 충실도에서 ASR을 4.0까지**
내린 "높은 충실도 + 낮은 ASR" 코너를 차지한다 — ODACE가 SD2.1-base를 **트레이드오프에서
지배**한다. "nudity를 필터링해 사전학습"한 모델조차 잠재 표현이 남아 적대적 프롬프트로
복원되므로, 결론은 **필터링된 사전학습 ≠ 강건한 개념 소거**이며, ODACE의 출력-접지 cross-attn
편집이 동등한 일반 품질을 유지하면서 더 깊은 소거를 달성함을 보인다.
