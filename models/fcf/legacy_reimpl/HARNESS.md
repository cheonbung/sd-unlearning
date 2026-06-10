# FCF 프로젝트 하네스 운영 가이드

하네스 엔지니어링이란: AI 에이전트가 같은 실수를 반복하지 않도록 **구조 자체로 방향을 잡아주는 것**.
"하지 말라"고 주의 주는 것이 아니라, 실수 자체가 불가능하거나 즉시 탐지되는 구조를 만드는 것.

---

## 파일 역할

| 파일 | 대상 | 역할 |
|------|------|------|
| `CLAUDE.md` | Claude (AI) | 매 대화마다 자동으로 로드되는 프로젝트 규칙 |
| `HARNESS.md` | 사람 (나) | 하네스 운영 방법 설명 (이 파일) |
| `scripts/validate.py` | Claude + 사람 | 코드 수정 후 실행하는 자동 검증기 |

---

## 하네스 운영 루틴

### Claude에게 작업 시키기 전

```
CLAUDE.md를 읽고 나서 작업해줘.
```
이 한 마디면 충분. Claude는 CLAUDE.md를 먼저 읽고 제약조건을 인식한 채 작업한다.

### Claude가 코드를 수정한 후

```bash
python scripts/validate.py
```
자동으로 다음을 체크:
- 필수 파일 존재 여부
- 논문 구현 규칙 위반 (UNet gradient, Stage 2 target 위치 등)
- Config 하이퍼파라미터 이탈
- Python 문법 오류

### 실수가 발생했을 때

1. 어떤 실수인지 파악
2. `CLAUDE.md`의 "절대 하지 말 것" 섹션에 한 줄 추가
3. 가능하면 `scripts/validate.py`에 자동 탐지 로직 추가
4. 다음엔 같은 실수가 구조적으로 차단됨

**이것이 하네스의 핵심:** 실수 → 규칙 추가 → 자동화의 사이클.

---

## CLAUDE.md 관리 원칙

- **60줄 이하로 유지** — 길어지면 Claude가 중요한 부분을 놓침
- **"현재 진행 상황" 섹션만 자주 업데이트** — 나머지는 안정적으로 유지
- 세부 내용은 별도 파일로 분리하고 CLAUDE.md에서 참조

### 자주 업데이트해야 하는 섹션

```markdown
## 현재 진행 상황 / 알려진 이슈
- [ ] 아직 안 된 것
- [x] 완료된 것
- ⚠️ 알려진 버그: 증상 → 원인
```

---

## Git 연동 (선택사항)

현재 이 프로젝트는 git repo가 아님. Git을 연동하면 pre-commit hook으로 validate.py를 자동 실행할 수 있음.

```bash
# Git 초기화
git init
git add .
git commit -m "initial commit"

# pre-commit hook 설정
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
echo "→ 하네스 검증 중..."
python scripts/validate.py --quick
if [ $? -ne 0 ]; then
    echo "✗ 검증 실패. 커밋 중단."
    exit 1
fi
EOF
chmod +x .git/hooks/pre-commit
```

이후 `git commit` 할 때마다 validate.py가 자동 실행되어 오류가 있으면 커밋이 차단됨.

---

## 확장 아이디어 (필요할 때)

- **실험 결과 트래커:** 각 실험 run의 하이퍼파라미터 + ASR 결과를 `outputs/results.jsonl`에 자동 기록
- **Config drift 탐지:** 논문 기준값에서 얼마나 벗어났는지 validate.py에서 더 엄격하게 체크
- **자동 플롯:** 학습 완료 후 `analysis/plot_training.py` 자동 실행
