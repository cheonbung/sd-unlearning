# Unlearning Methods Guide

이 문서는 본 실험의 표 A에 들어간 모든 모델이 **어디를 어떻게 바꾸는지**를 이해하기 쉽게 정리한다.
핵심 질문은 하나다.

> 같은 Stable Diffusion 계열 모델에서 target concept, 여기서는 주로 `nudity`, 를 지우기 위해
> 프롬프트 해석부, denoising 네트워크, 또는 추론 과정 중 어디를 건드렸는가?

최신 수치 표는 `compare/comparison_all_methods.md`를 기준으로 한다. 이 문서는 수치 순위보다
**unlearning 방식의 차이**에 집중한다.

## 1. 공통 생성 파이프라인

Stable Diffusion v1.x의 단순화된 흐름은 다음과 같다.

```text
prompt
  -> CLIP text encoder
  -> text embedding
  -> SD UNet denoising with cross-attention
  -> VAE decoder
  -> image
```

본 프로젝트의 방법들은 주로 세 지점을 건드린다.

| 개입 지점 | 무엇을 바꾸나 | 해당 모델 |
|---|---|---|
| CLIP text encoder | 프롬프트가 만드는 text embedding 자체를 바꿈 | FCF-P/E, Sph+OT, LSSE, DACE, Safe-CLIP |
| UNet | text embedding을 이미지 denoising으로 바꾸는 네트워크를 바꿈 | ESD-u, ODACE v2/v3/v1.5 |
| 추론 과정 | 가중치는 그대로 두고 sampling guidance만 바꿈 | safe_neg, SLD |
| 없음 / 사전학습 차이 | 실험 baseline | raw v1.4, raw v1.5, SD2.1-base |

비유로 보면 Stable Diffusion은 "주문서를 읽고 그림을 완성하는 제작 라인"이다.

```text
prompt               : 주문서
CLIP text encoder    : 주문서를 내부 작업 지시서로 번역하는 번역가
text embedding       : 번역된 작업 지시서
UNet                 : 작업 지시서를 보고 이미지를 실제로 조립하는 제작팀
cross-attention      : 작업 지시서가 제작팀 손에 들어가는 연결부
VAE decoder          : 마지막 출력 포맷으로 인쇄하는 후처리 장치
```

이 비유에서 각 방법은 서로 다른 위치를 고친다.

- text encoder 방법은 번역가의 습관을 바꾼다. 특정 주문어가 들어와도 내부 지시서에 그 의미를 덜 싣게 만든다.
- UNet 방법은 제작팀의 작업 방식을 바꾼다. 내부 지시서에 target concept가 남아 있어도 실제 조립 결과가 그쪽으로 가지 않게 만든다.
- 추론시 방법은 제작 라인은 그대로 두고, 매 주문마다 감독관이 옆에서 "이 방향은 피하라"고 조향한다.
- baseline 모델은 제작 라인을 고치지 않았을 때의 기본 행동을 보여준다.

## 2. 기호 읽는 법

아래 수식에서 쓰는 기호는 다음 뜻이다.

| 기호 | 뜻 |
|---|---|
| `p_f` | forget prompt. 지우고 싶은 concept를 포함한 프롬프트 |
| `p_n` | noise prompt. FCF에서 forget prompt가 향하도록 만든 무의미 문자열 |
| `p_r` | retain prompt. 보존해야 하는 일반 프롬프트 |
| `p_neu` | neutral prompt. explicit prompt에서 concept 단어를 제거한 프롬프트 |
| `T_theta(p)` | 학습 중인 CLIP text encoder가 prompt `p`를 embedding으로 바꾼 결과 |
| `T_0(p)` | frozen original CLIP text encoder의 embedding |
| `z_t` | diffusion timestep `t`의 latent |
| `eps_theta(z_t, t, c)` | UNet이 latent `z_t`와 conditioning `c`에서 예측한 noise |
| `c_f`, `c_r`, `c_0` | forget, retain, unconditional text conditioning |
| `MSE(a,b)` | 두 tensor를 같게 만들도록 하는 평균제곱오차 |

직관적으로,

- text encoder 방법은 `T_theta(p_f)`가 더 이상 nudity 의미를 담지 않게 만든다.
- UNet 방법은 `eps_theta(z_t,t,c_f)`가 nudity 이미지를 향하지 않게 만든다.
- 추론 방법은 학습 없이 sampling 중 안전 방향으로 조향한다.

## 3. 표 A 모델 전체 지도

| 모델 | 학습 여부 | 바뀌는 부분 | 한 줄 요약 |
|---|---:|---|---|
| raw v1.4 | 아니오 | 없음 | SD v1.4 원본 baseline |
| raw v1.5 | 아니오 | 없음 | SD v1.5 원본 baseline |
| SD2.1-base | 아니오 | 사전학습 데이터/필터 차이 | NSFW 필터링된 base지만 명시적 unlearning은 아님 |
| safe_neg | 아니오 | 추론 negative prompt | nudity 관련 단어를 negative prompt에 넣어 sampling 억제 |
| SLD-Medium/Strong/Max | 아니오 | 추론 safety guidance | denoising step마다 safety concept 방향을 빼는 guided sampling |
| Safe-CLIP | 아니오 | CLIP text encoder 교체 | 공개 Safe-CLIP text encoder로 SD text encoder를 교체 |
| ESD-u | 예 | UNet 비-cross-attn 위주 | frozen UNet의 negative guidance target으로 cross-attn을 제외한 UNet을 학습 |
| FCF-P | 예 | CLIP text encoder | explicit은 noise로, implicit은 concept projection을 제거한 target으로 이동 |
| FCF-E | 예 | CLIP text encoder | explicit은 noise로, implicit은 empirical concept vector를 빼는 target으로 이동 |
| Sph+OT | 예 | CLIP text encoder | FCF-P에 spherical projection과 OT-learned noise prompt를 결합 |
| vanilla LSSE | 예 | CLIP text encoder 일부 layer | concept null-space 제거 + contrastive retain + layer masking |
| LSSE +PLU | 예 | CLIP text encoder 일부 layer | LSSE에 progressive layer unlocking 추가 |
| LSSE +PLU+W2 | 예 | CLIP text encoder 일부 layer | PLU에 margin CNP를 더해 concept rerouting 억제 |
| DACE v2 | 예 | CLIP text encoder | explicit-neutral concept shift subspace를 동적으로 줄임 |
| DACE+PLU | 예 | CLIP text encoder 일부 layer | DACE에 progressive layer unlocking 추가 |
| ODACE v2 | 예 | UNet cross-attn K/V | output loss를 쓰지만 K/V-only 약한 편집 |
| ODACE v3 | 예 | UNet full cross-attn | output-grounded target으로 cross-attn q/k/v/out을 학습 |
| ODACE v1.5 | 예 | UNet full cross-attn | ODACE v3 recipe를 SD v1.5 base에 적용 |

## 4. Baseline Models: raw v1.4, raw v1.5, SD2.1-base

### raw v1.4 / raw v1.5

두 모델은 unlearning을 하지 않은 원본 Stable Diffusion baseline이다. ASR 표에서 이들은
"아무것도 지우지 않으면 공격 프롬프트가 얼마나 자주 성공하는가"를 보여준다.

비유: 제작 라인을 전혀 손보지 않은 상태다. 공격 prompt는 까다로운 주문서이고, raw baseline은
그 주문서를 원래 학습된 습관대로 그대로 처리한다. 따라서 raw ASR은 "소거 장치를 달기 전 기본 위험도"에 가깝다.

### SD2.1-base

SD2.1-base는 사전학습 단계에서 NSFW filtering의 영향을 받은 base model로 취급한다. 하지만 본 실험에서
별도 concept erasure를 학습한 것은 아니다.

중요한 차이는 다음과 같다.

- raw 모델: unlearning 없음.
- SD2.1-base: 사전학습 데이터/필터링 차이.
- FCF/LSSE/DACE/ODACE/ESD: 특정 target concept를 사후적으로 지우는 unlearning.

그래서 SD2.1-base는 "필터링된 사전학습이 강건한 unlearning을 대체할 수 있는가"를 보는 baseline 모델이다.

비유: 애초에 더 정제된 교재로 훈련받은 제작 라인이다. 하지만 특정 금지 주문을 받은 뒤 그 주문을
어떻게 거절해야 하는지 별도로 훈련받은 것은 아니다. 그래서 일반 품질은 좋을 수 있지만, 공격 prompt가
잠재적으로 남아 있는 표현을 다시 끌어내면 ASR이 높게 남을 수 있다.

## 5. safe_neg: negative prompt만 쓰는 추론시 억제

`safe_neg`는 학습하지 않는다. SD v1.5 pipeline에 고정 negative prompt를 넣는다.

```text
nudity, nude, naked, bare skin, breasts, nipples, genitals,
explicit, sexual, NSFW, erotic, pornographic
```

일반 classifier-free guidance를 아주 단순화하면 다음처럼 볼 수 있다.

```text
eps = eps_neg + s * (eps_text - eps_neg)
```

여기서

- `eps_text`는 사용자 prompt 방향의 noise prediction이다.
- `eps_neg`는 negative prompt conditioning에서 나온 prediction이다.
- `s`는 guidance scale이다.

일반 SD에서는 negative prompt가 비어 있는 경우가 많다. `safe_neg`는 unconditional 대신
nudity 관련 negative prompt를 넣어서, sampling trajectory가 nudity 방향으로 가는 것을 줄인다.

비유: 제작 라인을 고치지는 않고, 주문서 옆에 "이런 요소는 넣지 말 것"이라는 금지 메모를 붙이는 방식이다.
번역가와 제작팀은 그대로라서 빠르고 간단하지만, 주문서가 금지 메모를 교묘하게 우회하면 내부 습관은 여전히 남아 있다.

조금 더 풀어보면 `safe_neg`는 nudity 전용 금지 메모다. 그래서 nudity ASR에서는 생각보다 강하게 작동할 수 있지만,
이 힘은 모델이 target concept를 잊어서 생긴 것이 아니라 sampling 때마다 반대 조건을 넣어서 생긴다.

차별점:

- 장점: 학습 비용 0, 적용이 쉽다.
- 단점: 모델 내부 표현은 그대로라 적대적 prompt가 우회할 수 있다.
- 본 실험상 의미: 단순 추론 조향치고 ASR이 낮아 강한 baseline이지만, 진짜 unlearning은 아니다.

## 6. SLD: Safe Latent Diffusion

SLD도 학습하지 않는다. 대신 매 denoising step마다 세 가지 noise prediction을 계산한다.

```text
eps_uncond : unconditional prediction
eps_text   : 사용자 prompt prediction
eps_safety : safety concept prompt prediction
```

SLD의 핵심은 사용자 prompt 방향에서 safety concept 방향을 빼는 것이다.

```text
g_text = eps_text - eps_uncond
```

`g_text`는 "사용자 prompt가 unconditional 대비 latent를 어디로 끌고 가는가"를 뜻한다.

그 다음 safety 방향을 만든다.

```text
scale = clamp(|eps_text - eps_safety| * s_S, max=1)
safety_scale = where(eps_text - eps_safety >= lambda, 0, scale)
g_safety = (eps_safety - eps_uncond) * safety_scale + s_m * momentum
momentum := beta_m * momentum + (1 - beta_m) * g_safety
```

풀이:

- `eps_safety - eps_uncond`는 safety concept 쪽 방향이다.
- `s_S`는 safety guidance 강도다.
- `lambda`는 이미 충분히 safety concept에서 멀어진 위치에는 추가 조향을 줄이는 threshold다.
- `momentum`은 이전 step의 safety 조향을 이어받아 더 안정적으로 밀어내는 항이다.

최종 prediction은 다음처럼 만든다.

```text
if step >= warmup:
    g_text = g_text - g_safety

eps_pred = eps_uncond + guidance_scale * g_text
```

즉 "사용자 prompt를 따르되, safety concept 방향은 빼고 간다"는 방식이다.

비유: `safe_neg`가 주문서에 금지 메모를 붙이는 방식이라면, SLD는 제작 과정 매 단계마다 감독관이
중간 결과를 보고 "지금 안전하지 않은 방향으로 너무 가까워지고 있으니 조금 틀어라"라고 조향하는 방식이다.

조금 더 풀어보면 SLD는 한 번만 금지어를 넣는 것이 아니라 denoising step마다 safety concept prediction을
새로 계산한다. 그래서 계산량은 늘지만, 더 알고리즘적인 안전 조향을 한다. 다만 safety concept가 넓기 때문에
nudity 하나에만 최적화된 조향은 아니고, 공격 prompt가 강하면 감독관의 조향보다 주문서의 힘이 더 커질 수 있다.

표 A의 SLD 모델들은 같은 방법에 강도만 다르다.

| 모델 | warmup | safety scale `s_S` | threshold `lambda` | momentum |
|---|---:|---:|---:|---|
| SLD-Medium | 10 | 1000 | 0.01 | 0.3, beta 0.4 |
| SLD-Strong | 7 | 2000 | 0.025 | 0.5, beta 0.7 |
| SLD-Max | 0 | 5000 | 1.0 | 0.5, beta 0.7 |

차별점:

- safe_neg보다 일반 안전 concept가 넓다. nudity 전용이 아니라 violence, drugs, weapons 등까지 포함한다.
- 가중치는 바꾸지 않아 재사용이 쉽다.
- 공격 prompt가 text conditioning을 강하게 밀면 우회될 수 있다.

## 7. Safe-CLIP: 공개 안전 CLIP text encoder 교체

Safe-CLIP은 본 실험에서 새로 학습하지 않고 공개 checkpoint를 사용한다.

```text
기존 SD text_encoder <- Safe-CLIP의 fine-tuned text_model weights
UNet, VAE는 그대로
```

직관적으로는 prompt를 읽는 CLIP text encoder가 NSFW text/image embedding을 더 안전한 영역으로
redirect하도록 이미 학습되어 있고, 우리는 그 text encoder를 SD pipeline에 끼워 넣는다.

비유: 제작팀은 그대로 두고, 주문서를 번역하는 번역가만 안전 교육을 받은 번역가로 교체하는 것이다.
같은 prompt가 들어와도 제작팀에 전달되는 내부 작업 지시서가 달라진다.

조금 더 풀어보면 Safe-CLIP은 "현재 프로젝트에서 학습한 모델"이 아니라 "이미 안전 방향으로 fine-tune된
CLIP text encoder"를 가져와 SD에 맞게 끼운다. 따라서 실험에서는 학습 비용이 없지만, intervention의 깊이는
negative prompt보다 깊다. prompt embedding 자체가 바뀌기 때문이다.

차별점:

- 추론시 negative prompt보다 더 깊다. 프롬프트 embedding 자체가 바뀐다.
- 그래도 UNet은 그대로라 text encoder를 우회하는 적대 prompt에는 취약할 수 있다.
- 이 프로젝트의 FCF/LSSE/DACE와 같은 "text encoder 개입" 계열이지만, 본 실험에서는 학습이 아니라 공개 가중치 교체다.

## 8. ESD-u: Erased Stable Diffusion

ESD-u는 UNet을 직접 학습한다. CLIP text encoder는 frozen이다.

비유: 번역가는 그대로 두고 제작팀의 작업 습관을 고친다. 금지된 주문어가 포함된 작업 지시서가 들어와도,
제작팀이 그 요소를 조립하지 않도록 훈련한다.

핵심 아이디어는 concept prompt가 만든 denoising 방향을 반대로 누르는 것이다. frozen 원본 UNet으로 두 prediction을 만든다.

```text
eps_0 = frozen_UNet(z_t, t, unconditional)
eps_p = frozen_UNet(z_t, t, concept_prompt)
```

`eps_p - eps_0`는 "concept prompt가 unconditional 대비 denoising을 어느 방향으로 바꾸는가"를 뜻한다.
ESD는 target을 다음처럼 만든다.

```text
target = eps_0 - eta * (eps_p - eps_0)
```

풀이:

- `eps_0`는 concept가 없는 기준점이다.
- `eps_p - eps_0`는 concept가 추가한 방향이다.
- `- eta * (eps_p - eps_0)`는 그 concept 방향의 반대쪽으로 밀어내는 항이다.
- `eta=1`이면 concept 방향을 기준점 너머로 한 번 뒤집는 효과가 있다.

학습 loss는 다음과 같다.

```text
L = MSE(eps_trainable, target)
```

여기서 `eps_trainable`은 학습 중인 UNet이 concept prompt에서 예측한 noise다.

조금 더 풀어보면 ESD는 "concept prompt가 UNet 출력을 어떻게 바꾸는지"를 frozen UNet으로 먼저 측정한다.
그 다음 학습 중인 UNet에게 그 변화 방향을 그대로 따라가지 말고 반대로 움직이라고 시킨다. 그래서 text encoder를
고치는 방법보다 이미지 생성부에 더 가까운 intervention이다.

표 A의 `ESD-u`는 `noxattn` 설정, 즉 cross-attention을 제외한 UNet parameter를 크게 학습하는 canonical
nudity erasure 설정이다. 이름의 `u`는 nudity erasure에서 표준적으로 쓰인 ESD-u 설정을 가리키며,
본 프로젝트 구현에서는 `models/esd/esd_params.py`의 trainable mask가 실제 학습 범위를 결정한다.

차별점:

- text encoder가 아니라 UNet 가중치를 바꾸므로 추론/텍스트 개입보다 깊다.
- 많은 UNet parameter를 건드려 학습 비용과 침습성이 크다.
- 하지만 output을 직접 보지는 않고, frozen UNet의 negative guidance target을 따라가는 방식이다.

## 9. FCF: Fortified Concept Forgetting

FCF-P와 FCF-E는 모두 CLIP text encoder만 학습한다. UNet과 VAE는 frozen이다.
표 A의 FCF-P/E 행은 프로젝트 자체 `fcf/` 재구현 초깃값이 아니라, 공식 저자 코드와 저자 학습 데이터로
다시 학습한 reproduction checkpoint를 기준으로 한다.

비유: 제작팀은 그대로 두고 번역가를 재훈련한다. target concept가 들어간 주문서를 받으면, 번역가가
그 의미를 정상적인 작업 지시서로 번역하지 못하게 만든다. 대신 무의미한 지시서나 concept 성분이 제거된
지시서에 가깝게 번역하도록 학습한다.

FCF는 2단계 구조다.

### 9.1 Stage 1: explicit concept forgetting

문장 삼중쌍을 사용한다.

```text
p_f : explicit forget prompt
p_n : paired noise prompt
p_r : retain prompt
```

손실은 다음과 같다.

```text
L_forget = MSE(T_theta(p_f), T_0(p_n))
L_retain = MSE(T_theta(p_r), T_0(p_r))
L_total  = L_retain + eta * L_forget
```

풀이:

- `T_theta(p_f)`는 현재 text encoder가 forget prompt를 읽은 embedding이다.
- `T_0(p_n)`은 원본 encoder가 noise prompt를 읽은 embedding이다.
- 따라서 `L_forget`은 "nudity prompt를 noise처럼 읽게 만들라"는 뜻이다.
- `L_retain`은 retain prompt는 원래 embedding과 같게 유지하라는 뜻이다.

비유: 번역가에게 "이 금지 주문서는 의미 있는 작업 지시서가 아니라 잡음처럼 처리해라"라고 가르치는 단계다.
동시에 일반 주문서는 예전처럼 정확히 번역해야 하므로 retain loss를 같이 둔다.

### 9.2 Stage 2: implicit concept forgetting

explicit prompt만 지우면 `woman`, `man`, `body` 같은 암시적 표현으로 concept가 남을 수 있다.
FCF는 Stage 2에서 implicit concept embedding을 추가로 조정한다.

비유: explicit 단어만 금지어 사전에 넣으면, 비슷한 우회 표현으로 같은 이미지를 주문할 수 있다.
Stage 2는 그런 우회 표현이 내부 작업 지시서에 target concept를 다시 싣지 못하게 추가로 손보는 과정이다.

### 9.3 FCF-P: projection feature forgetting

FCF-P는 explicit concept embedding 평균에서 concept direction을 만든다.

```text
c = mean(T_0(explicit_concepts))
c_hat = c / ||c||
x = mean(T_0(implicit_concepts))
proj_c(x) = <x, c_hat> * c_hat
cleaned = x - mu_p * proj_c(x)
```

풀이:

- `c_hat`은 "nudity 방향"이라고 보는 단위 벡터다.
- `x`는 implicit concept들의 평균 embedding이다.
- `proj_c(x)`는 `x` 안에 들어 있는 nudity 방향 성분이다.
- `cleaned`는 그 성분을 뺀 target embedding이다.

그 다음 현재 encoder가 implicit concept를 이 target으로 보내도록 학습한다.

```text
L_forget = MSE(T_theta(implicit_group), cleaned)
L_retain = MSE(T_theta(retain_text), T_0(retain_text))
L_total  = L_retain + eta * L_forget
```

차별점:

- explicit은 noise로 보낸다.
- implicit은 "concept 방향 성분을 뺀 embedding"으로 보낸다.
- 공식 저자 코드 재현에서는 문장 삼중쌍 데이터와 projection 정규화가 중요했다.

비유: 작업 지시서 안에 특정 색의 잉크가 섞여 있다고 보고, 그 색 성분만 스포이트로 빼내는 방식이다.
FCF-P의 projection은 "이 embedding 안에 target concept 방향이 얼마나 섞였는가"를 재고 그 성분을 제거한다.

### 9.4 FCF-E: empirical feature forgetting

FCF-E는 projection 대신 explicit-minus-noise 평균 vector를 쓴다.

```text
v_emp = mean(T_0(p_f) - T_0(p_n))
target = T_0(implicit_group) - mu_e * v_emp
```

풀이:

- `T_0(p_f) - T_0(p_n)`은 "forget prompt와 noise prompt의 차이"다.
- 이 차이를 평균내면 concept를 나타내는 empirical vector라고 본다.
- implicit embedding에서 이 vector를 빼면 concept 성분이 약해질 것이라고 가정한다.

차별점:

- FCF-P는 기하학적 projection을 뺀다.
- FCF-E는 prompt pair에서 관찰된 경험적 차이 vector를 뺀다.
- 본 실험에서는 공식 FCF-P가 FCF-E보다 강했다.

비유: FCF-P가 이론적으로 추정한 concept 색 성분을 빼는 방식이라면, FCF-E는 실제 예시 쌍들을 보고
"금지 주문서와 noise 주문서가 평균적으로 이만큼 다르다"는 차이를 직접 측정해 빼는 방식이다.

## 10. Sph+OT: FCF-P의 geometry와 noise target을 개선

`Sph+OT`는 `models/novel/`의 가장 강한 FCF 확장이다. 두 아이디어를 결합한다.

```text
Sph+OT = N5 spherical/Riemannian projection + N6 OT noise prompts
```

비유: FCF가 번역가를 재훈련하는 방법이라면, Sph+OT는 그 재훈련에서 두 가지를 더 신중하게 고른다.
첫째, 작업 지시서 공간이 평평한 종이가 아니라 휘어진 지도라고 보고 그 지도 위에서 움직인다.
둘째, 금지 주문서를 보낼 "무의미한 목적지"도 아무 곳이나 찍지 않고, target concept와 충분히 먼 곳을 고른다.

### 10.1 N5: spherical projection

FCF-P는 embedding을 평평한 Euclidean vector로 보고 projection을 뺀다.

```text
cleaned = x - mu_p * proj_c(x)
```

Spherical variant는 CLIP embedding이 단순한 Euclidean 공간 전체가 아니라 좁은 cone/manifold에 놓여
있다고 보고, 단위 초구면 위에서 geodesic step을 한다.

절차는 다음과 같다.

```text
u_x = x / ||x||
u_c = c / ||c||
v = Log_{u_x}(u_c)
u_new = Exp_{u_x}(-mu_p * v)
cleaned = ||x|| * u_new
```

풀이:

- `u_x`와 `u_c`는 target과 concept를 길이 1인 점으로 본 것이다.
- `Log_{u_x}(u_c)`는 sphere 위에서 `u_x`에서 `u_c`로 향하는 접선 방향이다.
- `-mu_p * v`는 concept 방향의 반대로 이동한다는 뜻이다.
- `Exp`는 접선 공간에서 이동한 결과를 다시 sphere 위 점으로 되돌린다.
- 마지막에 원래 norm `||x||`를 곱해 embedding 크기 스케일을 복구한다.

차별점:

- Euclidean subtraction보다 CLIP embedding 분포를 덜 벗어나게 하려는 목적이다.
- 같은 FCF-P 구조를 유지하면서 cleaned target 계산만 바꾼다.

비유: 곡면 지도 위에서 한 지점에서 다른 지점의 반대 방향으로 이동해야 하는데, 평면에서 직선으로 빼면
지도 밖의 어색한 위치를 가리킬 수 있다. Spherical projection은 곡면 위를 따라 이동하듯 embedding manifold 위에서
concept 반대 방향으로 이동하려는 시도다.

### 10.2 N6: OT noise prompts

기본 FCF는 random 5-character noise prompt를 forget target으로 쓴다. Sph+OT는 이 noise prompt를
무작위로 두지 않고, explicit prompt embedding 분포에서 멀리 떨어진 후보를 고른다.

후보 noise string들을 만들고, frozen CLIP으로 embedding한다.

```text
Z_explicit = {T_0(p_f)}
Z_noise_candidate = {T_0(noise_candidate)}
```

각 후보는 Wasserstein distance 또는 sliced Wasserstein distance로 점수화한다.

```text
score(candidate) = W(Z_noise_candidate, Z_explicit)
```

점수가 높다는 것은 explicit nudity embedding 분포에서 멀다는 뜻이다.
상위 후보를 noise vocabulary로 저장하고 Stage 1의 `p_n`으로 사용한다.

차별점:

- "noise로 보내라"는 FCF 생각은 유지한다.
- 단, 아무 noise가 아니라 CLIP embedding 공간에서 concept와 멀리 있는 noise를 고른다.
- Spherical projection과 결합되어 text-encoder-only 방법 중 강한 성능을 냈다.

비유: 금지 주문서를 폐기함에 넣을 때, 아무 상자에나 넣는 것이 아니라 target concept와 가장 관련이 적은
보관함을 고르는 것이다. OT noise는 이 "멀리 떨어진 보관함"을 embedding 분포 거리로 찾는다.

## 11. LSSE: Layer-Selective Semantic Erasure

LSSE는 FCF와 달리 2-stage가 아니다. 하나의 training loop에서 explicit, implicit, retain을 같이 처리한다.
또한 random noise prompt를 쓰지 않는다.

비유: FCF가 "금지 주문서를 잡음처럼 번역하라"고 가르치는 방식이라면, LSSE는 번역가의 머릿속에서
target concept가 지나가는 통로를 찾아 그 통로를 막는 방식이다. 동시에 일반 주문서의 의미 구조는
망가지지 않도록 따로 붙잡아 둔다.

LSSE의 기본 구성은 세 가지다.

```text
N7 CNP: concept 방향 제거
N8 CSR: retain 의미 보존
N9 CLM: concept 관련 layer만 학습
```

### 11.1 CNP: Concept Null-Space Projection

먼저 frozen encoder로 explicit prompts를 embedding하고 SVD를 한다.

```text
Z = flatten(T_0(explicit_prompts))
c_dir = first principal direction of centered Z
```

`c_dir`는 explicit concept가 가장 크게 변하는 방향이다.

학습 중 forget 또는 implicit embedding이 이 방향으로 가지 않도록 한다.

```text
L_CNP = mean((flatten(T_theta(p)) dot c_dir)^2)
```

풀이:

- `dot` 값은 현재 embedding이 concept direction에 얼마나 걸쳐 있는지다.
- 제곱을 줄이면 그 방향 성분이 0에 가까워진다.
- noise target 없이 "concept 방향의 null-space로 밀어넣는다"고 볼 수 있다.

비유: 작업 지시서에 target concept가 보통 특정 방향의 화살표로 표시된다고 하자. CNP는 그 화살표 방향으로
남아 있는 성분을 계속 지워, 지시서가 그 방향을 가리키지 않게 만든다.

### 11.2 CSR: Contrastive Semantic Retention

retain prompt는 원본 의미를 유지해야 한다. LSSE는 MSE 대신 InfoNCE를 쓴다.

```text
q_i = normalize(mean_tokens(T_theta(p_r_i)))
k_i = normalize(mean_tokens(T_0(p_r_i)))
logits_ij = q_i dot k_j / tau
L_CSR = CrossEntropy(logits, labels=i)
```

풀이:

- `q_i`는 현재 encoder의 retain embedding이다.
- `k_i`는 원본 encoder의 같은 retain prompt embedding이다.
- 같은 prompt끼리는 가깝게, 다른 prompt와는 구분되게 만든다.
- 단순히 한 점으로 회귀하는 MSE보다 retain 공간의 구조를 살리려는 목적이다.

비유: 일반 주문서들을 모두 원래 자리 근처에 붙잡아 두되, 서로의 차이까지 유지하게 하는 방식이다.
단순 MSE가 "이 좌표로 돌아와"라면, CSR은 "같은 주문서는 같은 주문서답게, 다른 주문서와는 구분되게 남아 있어"에 가깝다.

### 11.3 CLM: CAP-guided Layer Masking

CLIP text encoder의 모든 layer를 학습하지 않고, concept에 인과적으로 중요하다고 판단된 layer만 연다.

```text
trainable_layers = top_k_layers_by_CAP_score
```

CAP 파일이 없으면 early layer를 여는 fallback을 쓴다. 이 프로젝트의 nudity CAP 분석은 초기 CLIP layer가
concept encoding에 강하게 관여한다는 가설을 줬다.

차별점:

- FCF는 전체 text encoder를 학습한다.
- LSSE는 일부 layer만 학습해 보존성과 효율을 노린다.

비유: 번역가의 모든 습관을 고치는 것이 아니라, target concept를 처리할 때 실제로 많이 쓰이는 몇 개의
작업 단계만 열어 수리하는 방식이다. 덜 관련된 단계는 잠가 두어 일반 번역 능력이 크게 흔들리지 않게 한다.

### 11.4 vanilla LSSE

표 A의 `vanilla LSSE`는 기본 LSSE다.

```text
L_total = alpha * L_CNP_explicit
        + beta  * L_CSR
        + gamma * L_CNP_implicit
```

풀이:

- explicit prompt의 concept direction을 줄인다.
- implicit prompt의 concept direction도 줄인다.
- retain prompt는 frozen encoder와 의미 구조를 맞춘다.

### 11.5 LSSE +PLU

PLU는 Progressive Layer Unlocking이다.

```text
초반: 1개 layer만 학습
중반: 3개 layer 학습
후반: 6개 layer 학습
```

직관:

- 처음부터 많은 layer를 열면 retain 구조가 크게 흔들릴 수 있다.
- 먼저 concept에 민감한 좁은 layer에서 시작하고, 필요할 때 점진적으로 열어준다.
- 본 실험에서 LSSE 성능을 크게 끌어올린 핵심 변형이다.

비유: 처음부터 전체 시스템을 뜯어고치지 않고, 작은 조정부터 시작해 필요할 때만 더 많은 부품을 연다.
초반에는 손상 범위를 줄이고, 후반에는 남은 concept 신호를 더 넓게 제거할 수 있다.

### 11.6 LSSE +PLU+W2

W2는 margin CNP 또는 weak orthogonal anchoring이다. 기본 CNP에는 한 가지 문제가 있다.

```text
concept direction c_dir만 0으로 만들면,
모델이 concept를 c_dir과 직교한 다른 방향으로 숨길 수 있다.
```

이것을 concept rerouting이라고 부른다. W2는 projection 제거에 더해 직교 성분도 원본에 약하게 묶는다.

```text
L_W2 = mean(proj_current^2)
     + lambda_orth * mean(||z_current_orth - z_frozen_orth||^2)
```

풀이:

- 첫 항은 기존 CNP처럼 concept direction 성분을 지운다.
- 두 번째 항은 concept direction을 제외한 나머지 의미 성분이 원본에서 너무 멀어지지 않게 한다.
- `lambda_orth`는 작게 둔다. 너무 크게 두면 아무것도 못 지우고, 너무 작으면 rerouting을 못 막는다.

차별점:

- PLU가 "어떤 layer를 언제 열지"의 개선이라면,
- W2는 "지운 concept가 다른 embedding 방향으로 도망가는 것"을 줄이는 손실 개선이다.

비유: 문 하나를 막았더니 같은 통로가 옆문으로 새어 나가는 상황을 막는 것이다. W2는 target 방향의 문을
닫으면서도, 방 전체의 배치를 원래와 너무 다르게 바꾸지 못하게 약하게 고정한다.

## 12. DACE: Dynamic Adversarial Concept Erasure

DACE는 LSSE에서 발견한 concept rerouting 문제를 더 직접적으로 다룬다. 하지만 여전히 CLIP text encoder만 학습한다.

비유: LSSE가 미리 찾은 통로를 막는 방식이라면, DACE는 학습 도중 계속 지도를 다시 그린다.
막아 둔 통로를 concept가 우회하면, 새로 생긴 우회 통로를 다시 찾아 막으려는 방식이다.

핵심은 forget prompt와 retain prompt를 비교하는 축이 아니라, explicit prompt에서 concept 단어를 제거한
neutral prompt와의 차이를 concept shift로 본다는 점이다.

```text
d_i = pool(T_theta(p_explicit_i)) - pool(T_theta(p_neutral_i))
```

여기서 `pool`은 token embedding을 평균내어 하나의 vector로 만드는 과정이다.

DACE는 이 shift vector들의 top-k SVD subspace를 동적으로 계산한다.

```text
U = top_k_svd({d_i})
```

`U`는 현재 모델이 concept를 표현하는 live subspace다. 학습 중 주기적으로 다시 계산한다.

손실은 다음과 같다.

```text
shift = pool(T_theta(p_explicit)) - pool(T_theta(p_neutral))
L_forget = || shift^T U ||^2
```

풀이:

- explicit prompt와 neutral prompt의 차이가 concept다.
- 그 차이가 live concept subspace `U` 안에서 0이 되도록 만든다.
- 즉 "concept 단어를 넣어도 embedding이 concept 방향으로 움직이지 않게 하라"는 뜻이다.

조금 더 풀어보면 DACE는 "nudity prompt와 일반 retain prompt가 다르다"가 아니라
"같은 문장에서 nudity 관련 단어를 넣었을 때 embedding이 어떻게 이동하는가"를 본다. 이 차이를 줄이면,
concept 단어가 들어와도 text encoder 내부 지시서가 그 concept 방향으로 변하지 않기를 기대한다.

DACE는 concept가 다른 방향으로 새어 나가지 않도록 orthogonal component도 묶는다.

```text
L_ortho = || orthogonal(pool(T_theta(p_explicit)))
           - orthogonal(pool(T_0(p_explicit))) ||^2
```

retain/neutral 보존은 다음처럼 한다.

```text
L_retain = MSE(T_theta(p_neutral), T_0(p_neutral))
         + MSE(T_theta(p_retain),  T_0(p_retain))
```

전체 손실:

```text
L_total = alpha * L_forget + gamma * L_ortho + beta * L_retain
```

### DACE v2

표 A의 `DACE v2`는 위 concept-axis DACE 기본형이다.

차별점:

- LSSE의 고정 concept direction보다 더 동적이다.
- 하지만 여전히 text embedding proxy를 최적화하므로, image-level ASR을 완전히 결정하지 못했다.

비유: 번역가의 내부 지시서만 검사하는 품질 관리다. 내부 지시서에서 target concept의 흔적이 줄어도,
제작팀인 UNet이 그 지시서를 이미지로 해석하는 방식까지 완전히 통제하는 것은 아니다.

### DACE+PLU

`DACE+PLU`는 DACE 손실에 progressive layer unlocking을 붙인 variant다.

```text
초반 1 layer -> 중반 3 layers -> 후반 6 layers
```

실험상 DACE+PLU는 오히려 ASR이 악화됐다. 이 결과는 "text embedding에서 concept 축을 잘 누르는 것"과
"최종 이미지에서 concept가 사라지는 것"이 항상 같지 않다는 음성 결과로 해석된다.

비유: 번역가의 여러 작업 단계를 점진적으로 열어 고쳤지만, 고친 내부 지시서가 제작팀에게는 오히려
다른 방식으로 해석될 수 있었던 사례다. 그래서 DACE+PLU는 "더 많이 또는 더 정교하게 text encoder를
고치는 것"이 항상 좋은 ASR로 이어지지는 않는다는 경고 역할을 한다.

## 13. ODACE: Output-Distribution Adversarial Concept Erasure

ODACE는 본 프로젝트의 중요한 전환점이다. text embedding proxy가 아니라 **UNet noise prediction 자체**를
학습 목표로 삼는다.

```text
prompt -> CLIP text encoder -> text embedding -> UNet -> noise prediction
                                               ^ ODACE edits here
```

CLIP text encoder는 frozen이다. UNet의 cross-attention projection만 학습한다.

비유: ODACE는 번역가의 내부 지시서를 고치는 대신, 제작팀이 실제로 이미지를 조립하는 손놀림을 고친다.
특히 작업 지시서가 제작팀에 전달되는 연결부, 즉 cross-attention을 고친다. 그래서 "지시서가 안전해 보이는가"보다
"최종 제작 동작이 target concept를 향하는가"를 직접 본다.

### 13.1 Output-grounded forget target

먼저 frozen UNet으로 concept trajectory의 latent `z_t`를 얻는다. 그 위치에서 두 prediction을 계산한다.

```text
eps_0 = frozen_UNet(z_t, t, c_0)
eps_p = frozen_UNet(z_t, t, c_f)
```

target은 다음과 같다.

```text
target = eps_0 - eta * (eps_p - eps_0)
```

이는 ESD와 같은 negative guidance 형태다. 다르게 쓰면 다음과 같다.

```text
target = eps_0 + eta * (eps_0 - eps_p)
```

풀이:

- `eps_p - eps_0`는 forget prompt가 image denoising에 추가한 concept 방향이다.
- `eps_0 - eps_p`는 그 반대 방향이다.
- `eta`를 크게 하면 concept 방향 반대로 더 강하게 민다.
- ODACE v3에서는 `eta=3.0`으로 강한 output-level erase target을 쓴다.

조금 더 풀어보면 `eps_p`는 "이 latent를 denoise하면 target concept가 있는 이미지 쪽으로 가라"는 제작팀의
동작에 가깝다. ODACE는 그 동작을 `eps_0` 근처로 되돌리는 데서 멈추지 않고, `eta`만큼 반대 방향으로 더 밀어
target concept가 강하게 억제되도록 만든다.

학습 중인 UNet의 forget prediction을 이 target에 맞춘다.

```text
L_forget = MSE(eps_trainable_forget, target)
```

retain prompt는 원본 UNet과 같게 유지한다.

```text
L_retain = MSE(eps_trainable_retain, eps_frozen_retain)
L_total  = alpha * L_forget + beta * L_retain
```

### 13.2 ODACE v2

ODACE v2는 UNet cross-attention의 K/V projection만 학습한 약한 편집이다.

```text
trainable: to_k, to_v
learning_rate: lower
eta: weaker
```

본 실험에서는 K/V만 조금 움직여서는 ASR이 raw에 가깝게 남았다. 그래서 v2는 중요한 negative result다.

비유: 제작팀과 작업 지시서의 연결부 중 일부 배선만 바꾼 것이다. 배선 일부를 고쳤지만 전체 동작이 충분히
달라지지 않아 target concept가 계속 살아남은 사례다.

### 13.3 ODACE v3

ODACE v3는 더 강한 recipe다.

```text
trainable: cross-attention to_q, to_k, to_v, to_out
learning_rate: 1e-4
num_steps: 1500
eta: 3.0
sample_guidance: 3.0
```

차별점:

- text embedding을 지우는 것이 아니라, text conditioning이 실제 denoising output에 미치는 효과를 줄인다.
- cross-attention은 text embedding이 UNet feature에 주입되는 지점이라 target concept 제거에 직접적이다.
- 일반 retain prompt에서는 frozen UNet prediction을 맞춰 locality를 보존한다.

비유: v3는 연결부 전체를 더 넓게 조정한다. cross-attention의 query, key, value, output projection을
함께 조정해 target concept 주문이 실제 제작 동작으로 이어지는 힘을 크게 낮춘다. retain loss는 일반 주문서가
여전히 원래 방식으로 제작되도록 잡아 주는 안전장치다.

### 13.4 ODACE v1.5

ODACE v1.5는 같은 full cross-attention recipe를 SD v1.5 base에 적용한 transfer check다.

의미:

- ODACE가 SD v1.4에만 우연히 맞은 recipe인지 확인한다.
- v1.5에서도 유사한 ASR을 보여 base 버전 간 전이가 확인됐다.

비유: 같은 수리법을 비슷하지만 다른 제작 라인에 적용해 본 것이다. v1.5에서도 잘 작동했다는 것은
ODACE의 수리 지점이 특정 체크포인트의 우연한 약점만 찌른 것이 아니라, SD v1.x 계열의 공통 구조와 맞닿아 있음을 시사한다.

## 14. 같은 이름처럼 보여도 중요한 차이

### FCF-P vs Sph+OT

둘 다 FCF-P 계열이지만 target 계산이 다르다.

```text
FCF-P:  Euclidean projection subtraction
Sph+OT: spherical geodesic step + OT-selected noise endpoint
```

FCF-P는 공식 저자 코드와 데이터로 재현했을 때 매우 강했다. Sph+OT는 프로젝트 내 독립 확장으로,
기하와 noise target 선택을 바꿔 text-encoder-only 한계를 더 밀어본 방법이다.

### LSSE vs DACE

둘 다 text encoder를 학습하고 concept direction/subspace를 줄인다.

```text
LSSE: frozen explicit embedding에서 고정 concept direction을 뽑음
DACE: 현재 explicit-neutral shift에서 live concept subspace를 반복 계산
```

LSSE는 layer 선택과 retain contrast가 강점이다. DACE는 rerouting을 추적하려 했지만, 이미지 ASR에서는
강한 결과로 이어지지 않았다.

### ESD vs ODACE

둘 다 UNet을 학습하고 negative guidance target을 쓴다.

```text
ESD-u: 넓은 UNet parameter를 canonical recipe로 편집
ODACE: cross-attention output behavior를 retain loss와 함께 직접 최적화
```

ODACE의 차별점은 "UNet 전체를 많이 건드리는 것"보다 "text conditioning이 image denoising으로 들어가는
cross-attention 지점을 output-grounded loss로 누르는 것"에 있다.

### safe_neg vs SLD

둘 다 학습하지 않는다.

```text
safe_neg: negative prompt 하나로 nudity를 직접 억제
SLD: safety concept prediction을 매 denoising step에서 계산해 빼는 알고리즘
```

safe_neg는 nudity에 특화되어 간단하고 강력한 편이다. SLD는 더 일반적인 safety concept를 다루지만,
본 실험의 nudity 공격에서는 Ring-A-Bell류에 취약했다.

## 15. 개입 지점별 핵심 결론

### 추론시 방법

대상 모델:

- safe_neg
- SLD-Medium
- SLD-Strong
- SLD-Max

특징:

- 학습 비용이 없다.
- 모델 가중치가 그대로라 배포가 쉽다.
- 하지만 적대 prompt가 원래 모델 내부 표현을 다시 끌어낼 수 있다.

### CLIP text encoder 방법

대상 모델:

- Safe-CLIP
- FCF-P
- FCF-E
- Sph+OT
- vanilla LSSE
- LSSE +PLU
- LSSE +PLU+W2
- DACE v2
- DACE+PLU

특징:

- prompt 해석 자체를 바꾸므로 추론시 방법보다 깊다.
- UNet은 그대로라, text embedding proxy가 image-level ASR을 완전히 결정하지 않는 경우가 있다.
- 공식 FCF-P와 Sph+OT처럼 강한 결과도 있지만, LSSE/DACE처럼 locality 또는 ASR에서 trade-off가 생긴 사례도 있다.

### UNet 방법

대상 모델:

- ESD-u
- ODACE v2
- ODACE v3
- ODACE v1.5

특징:

- 최종 이미지 생성에 가까운 denoising behavior를 바꾼다.
- ODACE v3/v1.5는 text encoder floor를 깨고 가장 낮은 ASR을 보였다.
- 다만 어떤 UNet parameter를 얼마나 여는지가 중요하다. ODACE v2의 K/V-only 약한 편집은 충분하지 않았다.

## 16. 상위 모델의 차별점과 기여

표 A에서 좋은 성능을 보인 `ODACE v3/v1.5`, `Sph+OT`, `LSSE+PLU+W2`는 모두 ASR을 낮췄지만,
기여의 성격은 서로 다르다. 쉽게 말해 세 방법은 "어디를 고칠 것인가"와 "무엇을 지웠다고 판단할 것인가"에
대해 다른 답을 낸다.

### 16.1 기존 방법의 네 부류

먼저 기존 방법을 네 부류로 나누면 차이가 더 잘 보인다.

| 부류 | 대표 모델 | 하는 일 | 핵심 한계 |
|---|---|---|---|
| 명시적 unlearning 없음 | raw v1.4/v1.5, SD2.1-base | 원본 모델 또는 필터링된 사전학습 모델을 그대로 사용 | 적대 prompt가 남은 표현을 다시 끌어낼 수 있음 |
| 추론시 조향 | safe_neg, SLD | sampling 중 negative prompt나 safety guidance로 nudity 방향을 억제 | 모델 내부 능력은 그대로라 우회 가능 |
| text encoder / text proxy 수정 | Safe-CLIP, FCF, Sph+OT, LSSE, DACE | prompt가 만드는 text embedding을 바꿈 | text embedding이 안전해 보여도 이미지 ASR이 완전히 결정되지는 않음 |
| UNet denoising behavior 수정 | ESD, ODACE | text conditioning이 실제 denoising으로 바뀌는 과정을 학습 | 학습 비용이 들고, 어떤 UNet 부분을 열지 중요 |

비유하면 raw와 SD2.1-base는 제작 라인을 그대로 쓰는 경우다. safe_neg와 SLD는 작업 중 옆에서 감독관이
말리는 방식이다. FCF/Sph+OT/LSSE/DACE는 주문서를 번역하는 번역가를 고치는 방식이다. ODACE는 번역된
작업 지시서가 제작팀 손에 들어간 뒤, 실제 조립 동작이 위험한 결과로 이어지지 않도록 제작팀의 핵심 연결부를
고치는 방식이다.

### 16.2 성능 위치

아래 표는 `compare/comparison_all_methods.md`의 표 A를 요약한 것이다. `ASR`은 낮을수록 좋다.

| 모델 | 계열 | 개입 지점 | ASR 8-lab↓ | 4-lab↓ | 해석 |
|---|---|---|---:|---:|---|
| raw v1.4 | baseline | 없음 | 62.0 | 46.4 | 아무것도 지우지 않았을 때의 위험도 |
| raw v1.5 | baseline | 없음 | 60.0 | 42.0 | v1.5 원본 비교 기준 |
| DACE v2 | text encoder | dynamic concept axis | 50.8 | 42.8 | text concept 축을 줄였지만 ASR 개선은 제한적 |
| Safe-CLIP | text encoder | encoder 교체 | 44.0 | 29.2 | 공개 안전 encoder 교체만으로는 공격에 취약 |
| SLD-Max | 추론시 | safety guidance | 45.2 | 18.0 | 강한 추론 조향이지만 Ring-A-Bell류에 취약 |
| ESD-u | UNet | 비-cross-attn UNet 편집 | 21.6 | 11.6 | UNet 편집이라 강하지만 ODACE보다는 얕음 |
| FCF-P | text encoder | 공식 FCF-P 재현 | 17.2 | 5.2 | 충실한 text encoder unlearning은 강할 수 있음 |
| LSSE+PLU+W2 | text encoder | concept null-space + layer 제어 | 20.8 | 7.2 | ASR 개선은 크지만 locality 대가가 큼 |
| Sph+OT | text encoder | spherical + OT noise | 15.6 | 1.6 | text encoder 계열 상위권 |
| ODACE v3 | UNet | full cross-attn output loss | 4.0 | 0.4 | 가장 낮은 ASR, 일반 COCO 품질도 raw 근처 |
| ODACE v1.5 | UNet | v1.5 base transfer | 4.0 | 1.2 | 같은 recipe가 SD v1.5에서도 유지됨 |

이 표에서 중요한 점은 단순 순위보다 방향성이다. `Sph+OT`는 FCF 계열의 text encoder 해법을 더 정교하게
만든 사례이고, `LSSE+PLU+W2`는 noise prompt 없이 concept 방향을 직접 지우려 한 사례다. `ODACE`는
text embedding proxy를 넘어서 실제 output behavior를 직접 학습한 사례다.

### 16.3 ODACE v3/v1.5: output-grounded UNet cross-attention 편집

ODACE의 기여는 "UNet을 학습했다"보다 더 구체적이다. ODACE는 text conditioning이 이미지 denoising으로
들어가는 `cross-attention` 지점을 열고, 그 출력 행동을 직접 바꾼다. 즉 "text embedding이 안전해 보이는가"가
아니라 "이 prompt를 넣었을 때 UNet의 noise prediction이 nudity 이미지를 향하는가"를 직접 다룬다.

핵심 손실은 다음처럼 읽을 수 있다.

```text
e_0 = frozen UNet의 unconditional prediction
e_p = frozen UNet의 forget prompt prediction

target = e_0 - eta * (e_p - e_0)

L_forget = MSE(e_forget_current, target)
L_retain = MSE(e_retain_current, e_retain_frozen)
```

`e_p - e_0`는 forget prompt가 unconditional 대비 이미지를 어느 방향으로 끌고 가는지를 뜻한다. ODACE는 그
방향을 지우는 데서 멈추지 않고 `eta`만큼 반대 방향으로 더 밀어낸다. retain prompt에 대해서는 원본 UNet의
prediction을 유지하게 해서 일반 생성이 크게 흔들리지 않도록 한다.

차별점:

- Safe-CLIP/FCF/Sph+OT/LSSE/DACE보다 output에 더 가깝다. text embedding이 아니라 image denoising을 직접 학습한다.
- ESD-u처럼 UNet을 만지지만, 넓은 UNet 전체보다 text가 이미지로 들어가는 cross-attention 연결부에 집중한다.
- ODACE v3는 SD v1.4에서의 주 결과이고, ODACE v1.5는 같은 recipe를 SD v1.5에 적용한 base-transfer 검증이다.
- v1.4와 v1.5 모두 ASR 4.0을 기록해, 특정 checkpoint에만 맞춘 우연한 편집이 아니라 SD v1.x 계열의 공통 구조를 겨냥했음을 보여준다.

단점과 주의점:

- Ring-A-Bell처럼 prompt 의미 대부분이 nudity trigger와 얽힌 공격에서는 output이 원래 prompt와 크게 달라질 수 있다.
- 하지만 COCO-FID/CLIP에서는 raw와 가까워, 일반 prompt 전체가 망가진 것으로 보기는 어렵다.
- GPU 학습과 UNet checkpoint 관리가 필요하므로 safe_neg 같은 추론시 방법보다 운용 비용이 높다.

비유: 기존 text encoder 방법이 "주문서를 안전하게 번역하자"라면, ODACE는 "제작팀이 그 주문서를 받았을 때
위험한 손놀림으로 이어지는 연결부를 직접 고치자"에 가깝다.

### 16.4 Sph+OT: FCF의 이동 경로와 forget 목적지를 개선

Sph+OT는 완전히 새로운 파이프라인이라기보다, FCF-P의 약한 지점을 정교하게 바꾼 확장이다. FCF-P는 implicit
prompt embedding에서 concept 방향을 Euclidean projection으로 빼고, explicit forget prompt는 random noise
prompt 쪽으로 보낸다. Sph+OT는 이 두 선택을 각각 바꾼다.

```text
FCF-P  : x - eta * proj_c(x)
Sph+OT : Exp_x(-eta * Log_x(c)) + OT-selected noise endpoint
```

첫 번째 기여는 `Sph`, 즉 spherical/Riemannian 이동이다. CLIP embedding은 아무 방향으로나 퍼진 평평한 공간이라기보다
좁은 manifold나 cone 위에 모여 있다고 볼 수 있다. 그래서 concept 성분을 직선으로 빼면 UNet이 익숙하지 않은
OOD embedding으로 밀릴 수 있다. Sph+OT는 구면 위에서 concept 반대 방향으로 이동해, text embedding을 더 자연스러운
위치에 남기려 한다.

두 번째 기여는 `OT`, 즉 Optimal Transport로 고른 noise prompt다. 기본 FCF는 random 5-character 문자열을
forget target으로 쓰지만, 그 문자열이 embedding 공간에서 좋은 목적지라는 보장은 없다. Sph+OT는 후보 noise prompt들을
CLIP으로 인코딩한 뒤, explicit prompt 분포와 Wasserstein 거리가 큰 후보를 고른다. 즉 아무 보관함에 버리는 대신,
target concept와 충분히 멀리 떨어진 보관함을 골라 보내는 방식이다.

차별점:

- FCF-P보다 geometry를 더 존중한다. 직선으로 빼는 대신 구면 위 이동을 쓴다.
- random noise target 대신 OT로 선택한 noise endpoint를 쓴다.
- text encoder만 바꾸므로 UNet을 학습하는 ODACE보다 가볍고, FCF 계열 안에서는 강한 성능을 냈다.

단점과 주의점:

- 여전히 text embedding proxy 방법이다. UNet이 그 embedding을 실제 이미지로 어떻게 해석할지는 완전히 통제하지 않는다.
- retain CLIP이 일부 하락할 수 있고, output-grounded 방법인 ODACE만큼 낮은 ASR에는 도달하지 못했다.

비유: FCF-P가 금지 물건을 창고 밖으로 직선으로 밀어내는 방식이라면, Sph+OT는 실제 복도 구조를 따라 이동하고,
도착지도 아무 상자가 아니라 금지 물건과 가장 멀리 떨어진 보관함으로 고르는 방식이다.

### 16.5 LSSE+PLU+W2: noise prompt 없이 concept 통로를 지우는 text encoder 편집

LSSE의 기여는 FCF의 random noise prompt를 제거하고, text encoder 내부의 concept 방향을 직접 지우려 한 점이다.
explicit prompt embedding들에서 concept direction `c`를 찾고, 현재 embedding이 그 방향으로 갖는 성분을 줄인다.

```text
L_CNP = mean((z · c)^2)
```

`PLU`는 Progressive Layer Unlocking이다. 처음부터 많은 layer를 열면 text encoder가 크게 흔들릴 수 있으므로,
처음에는 적은 layer만 학습하고 이후 1개, 3개, 6개처럼 점진적으로 학습 범위를 넓힌다. 이 실험에서는 PLU가
vanilla LSSE의 ASR을 크게 낮춘 핵심 요인이었다.

`W2`, 즉 margin CNP는 concept 방향만 지우다가 의미가 다른 직교 방향으로 새어 나가는 문제를 줄이려는 장치다.

```text
L_W2 = concept projection 제거 + orthogonal component 보존
```

첫 항은 `nudity` 방향 성분을 줄인다. 두 번째 항은 concept와 무관한 나머지 의미가 원본 embedding에서 너무 멀어지지
않도록 잡아 준다. 즉 금지된 통로를 막되, 다른 정상 통로까지 같이 무너뜨리지 않으려는 설계다.

차별점:

- FCF처럼 noise prompt로 보내지 않고, concept null-space를 직접 만든다.
- CSR로 retain prompt의 관계를 보존하고, CLM/PLU로 어떤 layer를 얼마나 열지 제어한다.
- W2는 concept rerouting, 즉 concept가 다른 방향으로 새어 나가는 현상을 줄이려는 보강이다.

단점과 주의점:

- ASR은 raw 62.0에서 20.8까지 크게 낮아졌지만, COCO-FID 144.59, COCO-CLIP 19.19, LPIPS 0.609로 locality 손상이 크다.
- 따라서 LSSE+PLU+W2는 "text encoder만으로 꽤 강한 소거가 가능하다"는 기여와 동시에, "강한 text encoder 편집은 일반 생성 드리프트를 부를 수 있다"는 경고를 함께 보여준다.

비유: LSSE+PLU+W2는 번역가에게 이상한 무의미 문장으로 번역하라고 가르치는 대신, 번역가 머릿속의 금지 개념 통로를
찾아 막는다. PLU는 한 번에 모든 방을 뜯어고치지 않고 조금씩 문을 여는 방식이고, W2는 막은 통로 옆으로 새 길이
생기지 않도록 주변 구조를 붙잡는 장치다.

### 16.6 차별점 한눈에 보기

| 기존 한계 | Sph+OT의 답 | LSSE+PLU+W2의 답 | ODACE의 답 |
|---|---|---|---|
| random noise prompt가 좋은 forget 목적지인지 불명확 | OT로 concept 분포에서 먼 noise endpoint 선택 | noise prompt 자체를 제거 | forget 목적지를 text가 아니라 UNet output target으로 정의 |
| Euclidean projection이 CLIP embedding geometry를 무시할 수 있음 | spherical geodesic으로 자연스러운 이동 | concept null-space projection 사용 | embedding geometry보다 denoising behavior를 직접 최적화 |
| text encoder를 너무 많이 바꾸면 retain 의미가 흔들림 | FCF 구조 유지로 변경 범위 제한 | CSR, CLM, PLU, W2로 layer와 retain을 제어 | retain loss로 일반 denoising prediction을 원본에 고정 |
| text proxy가 image ASR을 완전히 설명하지 못함 | text proxy 안에서 최대한 개선 | text proxy의 한계를 실험적으로 드러냄 | proxy를 버리고 output-grounded loss로 이동 |
| 적대 prompt가 기존 safety 방법을 우회함 | Ring-A-Bell(Re)에 강한 text encoder 개선 | ASR을 낮추지만 locality 비용이 큼 | Ring-A-Bell 0.0, RaB(Re) 0.0으로 가장 강한 출력 수준 소거 |

정리하면 `Sph+OT`는 FCF 계열을 더 정교하게 만든 기여, `LSSE+PLU+W2`는 noise prompt 없는 concept-direction
소거의 가능성과 한계를 동시에 보여준 기여, `ODACE`는 text proxy를 넘어 output-grounded UNet editing으로
넘어간 기여다. 세 방법 모두 기존 모델보다 나은 지점을 만들었지만, 최종적으로 ASR과 locality를 함께 보면
ODACE v3/v1.5가 가장 강한 결론을 만든다.

## 17. 짧은 요약

| 계열 | 쉽게 말하면 | 강점 | 약점 |
|---|---|---|---|
| raw / SD2.1-base | 지우지 않은 baseline | 비교 기준 제공 | 공격에 취약 |
| safe_neg | "이런 것은 그리지 마"를 negative prompt로 넣음 | 즉시 적용 | 내부 표현은 그대로 |
| SLD | sampling 중 safety 방향을 계속 빼며 생성 | 무학습, 알고리즘적 조향 | 적대 prompt에 취약 |
| Safe-CLIP | 안전하게 fine-tune된 text encoder로 교체 | 간단한 encoder swap | UNet은 그대로 |
| FCF | text encoder가 concept prompt를 noise/cleaned feature처럼 읽게 함 | 공식 FCF-P는 강함 | 데이터/공식 구현 세부가 중요 |
| Sph+OT | FCF의 geometry와 noise target을 개선 | text encoder 계열 강한 성능 | 여전히 text proxy |
| LSSE | concept direction을 null-space로 지우고 layer를 선택 학습 | noise prompt 제거, PLU 효과 큼 | locality 손상 가능 |
| DACE | live concept shift subspace를 동적으로 지움 | rerouting을 직접 겨냥 | image ASR로 잘 이어지지 않음 |
| ESD | UNet을 negative guidance target으로 학습 | text/추론보다 깊은 편집 | 많은 UNet parameter 편집 |
| ODACE | UNet cross-attention output을 직접 concept 반대 방향으로 학습 | 가장 output-grounded, 강한 ASR | GPU 학습 필요 |

## 18. 어디를 보면 되나

| 주제 | 주요 파일 |
|---|---|
| 전체 수치 비교 | `compare/comparison_all_methods.md` |
| cross-model 평가 registry | `eval/xeval.py` |
| FCF 프로젝트 구현 | `fcf/core/trainer.py` |
| 공식 FCF 재현 산출물 | `models/fcf/official_fcf_p/`, `models/fcf/official_fcf_e/` |
| Sph+OT 구현 | `models/novel/methods/trainer.py`, `spherical.py`, `ot_noise.py` |
| LSSE 구현 | `models/lsse/methods/lsse_trainer.py`, `cnp.py`, `csr.py`, `clm.py` |
| DACE 구현 | `models/dace/core/trainer.py`, `models/dace/methods/erasure.py`, `adversary.py` |
| ODACE 구현 | `odace/core/trainer.py`, `odace/methods/unet_edit.py` |
| ESD 구현 | `models/esd/esd_trainer.py` |
| SLD 구현 | `models/sld/sld_pipeline.py` |
| Safe-CLIP loader | `models/safeclip/safeclip_loader.py` |
