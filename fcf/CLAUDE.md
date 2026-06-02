# FCF Project — Claude 컨텍스트 파일

## 이 프로젝트가 무엇인가

"Fortified Concept Forgetting (FCF)" 구현체.
SD의 **CLIP 텍스트 인코더만** 파인튜닝하여 특정 개념(nudity, violence, 스타일)을 잊게 만드는 machine unlearning 연구. UNet/VAE는 건드리지 않는다.

---

## 핵심 하이퍼파라미터 (임의 변경 금지)

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| `lr` | 2.5e-5 | CLIP 인코더 학습률 |
| `eta` (η) | 0.25 | Stage 2 이동 강도 |
| `mu_p` (μ_p) | 0.7 | FCF-P 임계값 |
| `mu_e` (μ_e) | 1.0 | FCF-E 임계값 |
| Stage 1 epochs | 60 | explicit forgetting |

---

## 알고리즘 구현 규칙 (절대 바꾸지 말 것)

1. **Stage 2 target은 학습 전 1회만 계산** — 루프 안에서 재계산 금지
2. **FCF-P projection 공식:** `target - η · proj` (부호 주의)
3. **FCF-E experience vector:** `mean(T_ori(P_explicit) - T_ori(P_noise))` — retain이 아닌 noise 프롬프트 (Eq. 7)
4. **노이즈 텍스트:** 5글자, 문자 반복 없음, 심볼+알파벳+숫자 혼합 (`noise_utils.py`)
5. **UNet, VAE 학습 금지** — CLIP 텍스트 인코더만 수정

---

## 절대 하지 말 것

- `outputs/` 체크포인트 삭제 — 재현성 소멸
- `evaluate.py` ASR 계산 로직 임의 수정 — 논문 비교 불가
- UNet 파라미터에 gradient 흘리기
- Stage 1/2 동시 실행 또는 순서 변경
- 논문에 없는 loss term 추가 (제안은 가능, 기본 config 제외)

---

## 자주 쓰는 커맨드

> **PowerShell 환경 — 줄 연속 문자는 `\` 아닌 `` ` ``. 혼동 방지를 위해 항상 한 줄로 작성.**
> (`\` 사용 시 `unrecognized arguments: \` 오류 발생)

```powershell
# 학습
python train.py --config configs/nudity_fcf_p.yaml
python train.py --config configs/nudity_fcf_p.yaml --stage1_steps 10 --stage2_steps 10

# 평가 / 이미지 생성
python evaluate.py --encoder_dir outputs/fcf_p_nudity/final --concept nudity --eval_type asr
python generate_images.py --encoder_dir outputs/fcf_p_nudity/final --prompts_file data/eval/i2p_nudity.txt --output_dir outputs/images/test

# Multi-GPU — rank별 인터리브 방식(파일명 충돌 없음)
python generate_images.py --prompts_file <file> --output_dir <dir> --rank 0 --world_size 2 --device cuda:0
python generate_images.py --prompts_file <file> --output_dir <dir> --rank 1 --world_size 2 --device cuda:1

# conda run 오류 시 → 직접 Python 경로 사용
c:\Users\vip\anaconda3\envs\fcf\python.exe train.py --config configs/nudity_fcf_p.yaml
```

---

## 구현 주의사항 (디버깅으로 확인된 사항)

### NudeNet v3 레이블 — 이전 버전과 완전히 다름

| 구버전 (잘못됨) | v3 (정확) |
|---|---|
| `EXPOSED_BREAST_F` | `FEMALE_BREAST_EXPOSED` |
| `EXPOSED_GENITALIA_F` | `FEMALE_GENITALIA_EXPOSED` |
| `COVERED_BREAST_F` | `FEMALE_BREAST_COVERED` |
| `EXPOSED_ANUS` | `ANUS_EXPOSED` |
| `EXPOSED_BUTTOCKS` | `BUTTOCKS_EXPOSED` |

`asr_evaluator.py`의 `NUDITY_LABELS` v3 형식으로 수정 완료. ASR 임계값: score > 0.3 (line 85)

### 환경 제약

- **FID on Windows:** `cleanfid` multiprocessing 오류 → `num_workers=0` 고정 (`fid_clip_evaluator.py` 수정 완료)
- **conda run 오류:** `conda run -n fcf` 사용 불가 → 직접 경로 `c:\Users\vip\anaconda3\envs\fcf\python.exe` 사용
- **logs/ 디렉토리:** 백그라운드 리다이렉션 전 수동 생성 필요

### 평가 공정성

- **Ring-A-Bell(Re):** 각 모델을 타겟으로 최적화 → Baseline/FCF-P/FCF-E 프롬프트 파일이 서로 다름
- **Violence I2P ≡ UDA:** 두 파일 내용 동일(마지막 개행만 다름) → ASR 수치 같은 것이 정상
- **LPIPS 평가:** `--baseline_other_dir` 명시 필요 (없으면 LPIPS_m=0 오류)

---

## 코드 수정 시 체크리스트

1. 논문의 어느 Algorithm에 해당하는 변경인가?
2. 다른 config (nudity/vangogh/violence)에 영향을 주는가?
3. `outputs/` 하위의 기존 결과가 invalidate되는가?

---

## 실험 결과

@docs/results.md
