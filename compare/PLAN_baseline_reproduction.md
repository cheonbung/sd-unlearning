# 실행 계획서 — 레퍼런스 비교모델 재현 (ESD · SLD · Safe-CLIP)

> **상태:** 계획 확정(사용자 승인). **실행은 다음 세션.** 이 문서 + `[[baseline-reproduction-plan]]`
> 메모리가 핸드오프 매개체. 새 세션은 MEMORY.md → 이 문서를 읽고 그대로 실행하면 된다.
> 원 대화 전체 기록(필요시): `C:\Users\USER\.claude\projects\d--unlearning-SD-unlearning\60aaedb5-b018-434a-bef1-277490b2f272.jsonl`

## 확정된 범위 (사용자 승인)
- **재현 대상:** ESD(학습) · SLD(재구현, 무학습) · Safe-CLIP(공개가중치, 무학습)
- **깊이:** 추론/공개가중치 우선, **학습은 ESD만**(단일 시드 먼저)
- **제외:** Gong(열역학)·Huang(대장암) = 도메인 무관 오분류
- **FCF:** 이미 재측정 보유(FCF-P 52.8 / FCF-E 61.2) → 원논문 보고값만 표 B에 추가
  > ⚠️ **정정(2026-06):** 위 52.8/61.2는 *우리 `fcf/` 재구현*(불충실)값이다. 이후 **저자 공식 코드+데이터로 재현**한 결과
  > FCF-P는 논문정렬 full-set 4-label ASR **3.7 ≈ 논문 3.43**(8-label 16.9), FCF-E 18.6으로 재현 확정. 상세: `comparison_all_methods.md` §③-정정.
- **base:** Rombach LDM = raw 기준선(방법 아님)

## 레퍼런스 분류 (`fcf/reference/`)
| 파일 | 정체 | 처리 |
|---|---|---|
| [2023] Gandikota — Erasing Concepts | **ESD** | 재현(학습) |
| [2023] Schramowski — Safe Latent Diffusion | **SLD** | 재현(가이던스 재구현) |
| [2025] Poppi — Safe-CLIP | **Safe-CLIP** | 재현(공개 텍스트인코더 교체) |
| [2026] Fan — Fortified Concept Forgetting | **FCF** | 보유, 원논문값만 |
| [2022] Rombach — LDM | SD base | raw 기준선 |
| [2024] Gong / [2024] Huang | 무관 | **제외** |

## 측정 프로토콜 (전 모델 동일 — 기존 계승)
- **ASR↓:** NudeNet v3(score>0.3), 5공격(I2P, Ring-A-Bell, RaB(Re), P4D, UnlearnDiffAtk)×50장, guidance 7.5·50 steps. `xmodel/xeval.py`.
- **COCO-FID/CLIP:** `xmodel/eval_coco.py`(COCO 300캡션 vs 실제 600장).
- **as-reported(표 B):** 각 논문 원프로토콜 그대로, 표 A와 **병합 금지**.

## 미러할 패턴
- 등록/파이프: `xmodel/xeval.py` `REGISTRY`(L81) + `build_pipe`(L117, `kind=="odace"` UNET swap 분기).
- 학습기: `odace/core/trainer.py`(ESD-스타일 음성가이던스, **FCFTrainer 비상속**), config `odace/configs/nudity_odace.yaml`.
- 평가: `xeval.run` + `eval_coco.run` → `metrics.json` + `coco_metrics.json`.
- 테스트: `xmodel/tests/test_xeval.py`.

## Phase별 작업

### Phase 0 — 레퍼런스 정밀 분석 (읽기 전용)
- 각 PDF(ESD/SLD/Safe-CLIP/FCF)에서 **원논문 보고 지표** 추출: nudity/I2P 비율, NudeNet exposed 개수, FID(레퍼런스셋), CLIP **+ 프로토콜**(공격셋·NudeNet 버전·FID refset). → 표 B 초안.
- 재현 사양 확정: ESD-u vs ESD-x / SLD config(Weak·Medium·Strong·Max) / Safe-CLIP HF 체크포인트 가용성·차원 일치.

### Phase 1 — Harness 확장 (`xmodel/xeval.py`)
- `REGISTRY`에 `esd`(unet_dir, kind="esd"=odace와 동일 swap), `sld`(kind="sld", config 파라미터), `safeclip`(kind="safeclip", text_encoder id) 추가.
- `build_pipe` 분기 추가:
  - `esd`: odace와 동일하게 `pipe.unet = UNet2DConditionModel.from_pretrained(unet_dir)`.
  - `sld`: 커스텀 SLD 가이던스(아래) 사용 플래그.
  - `safeclip`: `pipe.text_encoder = CLIPTextModel.from_pretrained(safeclip_id)` 교체(차원 768/77 검증).
- `xmodel/tests/test_xeval.py`에 신규 kind 등록 검증 테스트.

### Phase 2 — ESD 재현 (최우선, 학습)
- 신규 모듈 `baselines/esd/`(또는 `odace/` 내 `train_esd.py`). **공식 recipe**(Gandikota): 음성가이던스 타깃 `ε_θ(x,∅) − η(ε_θ(x,c)−ε_θ(x,∅))`, frozen 원본이 가이던스 제공, UNET만 학습.
  - **ESD-u**(cross-attn 제외 = unconditional, NSFW에 표준): η=1, lr 1e-5, ~1000 step.
  - **ESD-x**(cross-attn만): 개념 국소 소거.
  - ⚠️ ODACE(강화판)와 **구분** — 캐논 ESD recipe 그대로(ODACE의 lr 1e-4/eta 3.0 아님).
- 체크포인트 → `outputs/esd_u/final` → REGISTRY 등록.

### Phase 3 — SLD 재현 (무학습)
- diffusers 0.38이 `StableDiffusionPipelineSafe` 제거 → **가이던스 항 직접 구현**: 매 step ε(prompt)·ε(∅)·ε(safety concept) 계산 후 safety 방향에서 momentum/threshold로 멀어지기. 파라미터: warmup·guidance_scale·threshold·momentum_scale·mom_beta.
- config: SLD-Medium·Strong·Max 등록. (safe_neg를 진짜 SLD로 격상)
- 대안: 별도 env에 구 diffusers 핀(`StableDiffusionPipelineSafe`). 재구현 우선.

### Phase 4 — Safe-CLIP 재현 (무학습)
- 공개 가중치(`aimagelab/safeclip_vit-l_14` 가용성 Phase 0 확인) 텍스트인코더를 SD v1.x `pipe.text_encoder`에 교체. 추론 전용.

### Phase 5 — 통합 평가
- 신규 baseline 전부 `xeval`(ASR) + `eval_coco`(COCO) 동일 프로토콜 측정.
- `compare/comparison_all_methods.md` 표 A에 행 추가, `xmodel/build_xgallery.py` MODELS에 열 추가 후 갤러리 재생성.

### Phase 6 — 다방면 분석
- **표 A**(우리 harness, ASR+COCO) / **표 B**(원논문 보고값, 프로토콜 명시) 분리.
- 분석 축: ①공격별 ASR ②COCO 국소성 ③재현 충실도(우리값 vs 원논문값) ④개입지점 분류(텍스트인코더/UNET/추론/CLIP인코더/사전학습필터) ⑤연산비용(학습시간·편집 파라미터 수).

## 리스크
| 리스크 | 완화 |
|---|---|
| SLD 파이프라인 부재 | 가이던스 수동 재구현(우선) / 구 diffusers env 핀 |
| ESD 학습 비용·편차 | 공식 하이퍼파라미터, 단일 시드 먼저 |
| Safe-CLIP 가중치·호환성 | Phase 0에서 HF 가용성+차원 검증 |
| 원논문 프로토콜 상이 | 표 A/B 분리·프로토콜 명시 |
| GPU 비용 | ESD만 학습, 단계별 승인 |

## CLAUDE.md 제약 (필수)
- 부모 `fcf/`·`train_fcf.py`·`evaluate.py` 수정 금지(가중치/체크포인트 읽기는 OK).
- `fcf-novel-methods/` 코드 import 금지.
- 결과는 프로젝트 자체 `outputs/`에만.
- 신규 트레이너 **FCFTrainer 비상속**.
- NudeNet v3 score>0.3 · FID num_workers=0 · tmux 실행은 `tee`로 라이브 출력.

## 실행 환경
- WSL conda env python: `/home/user/miniconda3/envs/lsse/bin/python`
- 예: `wsl bash -lc 'cd /mnt/d/unlearning/SD_unlearning && /home/user/miniconda3/envs/lsse/bin/python xmodel/xeval.py --models esd_u'`
- 장기 실행: tmux + `tee`, 감시는 리터럴 경로 `until grep -qE "MARKER|Traceback" /path; do sleep 30; done`.

## 복잡도: HIGH (GPU 학습 포함)
