"""aggregate.py — LSSE 실험 결과 집계 (Phase 0 재현성 척추).

수동 cat-JSON을 자동화. outputs/eval/*/eval_results.json(ASR/quality)와
선택적으로 outputs/*/history.json(최종 진단)을 읽어 markdown + CSV 표 생성.

멀티시드: 실행명이 '<name>_seed<N>' 패턴이면 그룹화하여 mean±std 계산.

Usage:
    python experiments/aggregate.py --eval_root outputs/eval --out outputs/summary.md
    python experiments/aggregate.py --eval_root outputs/eval --hist_root outputs --out outputs/summary.md

ASR 낮을수록 좋음(망각 강함). FID 낮을수록·CLIP 높을수록 좋음(품질 보존).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ATTACKS = ["I2P", "Ring-A-Bell", "Ring-A-Bell(Re)", "P4D", "UnlearnDiffAtk"]
SEED_RE = re.compile(r"_seed\d+$|_s\d+$")


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.0f}" if pct else f"{v:.2f}"


def load_eval_dir(eval_root: str) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    root = Path(eval_root)
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        f = d / "eval_results.json"
        if f.exists():
            try:
                rows[d.name] = json.loads(f.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                pass
    return rows


def extract_asr(data: dict) -> Tuple[Dict[str, Optional[float]], Optional[float]]:
    asr = data.get("asr", {}) or {}
    vals = {a: (asr.get(a, {}) or {}).get("asr") for a in ATTACKS}
    present = [v for v in vals.values() if v is not None]
    mean = sum(present) / len(present) if present else None
    return vals, mean


def extract_quality(data: dict) -> Tuple[Optional[float], Optional[float]]:
    q = data.get("quality") or {}
    fid = q.get("fid", q.get("FID"))
    clip = q.get("clip_score", q.get("clip", q.get("CLIP_Score", q.get("CLIP"))))
    return fid, clip


def load_final_diag(hist_root: Optional[str], run_name: str) -> Dict[str, float]:
    if not hist_root:
        return {}
    # eval 이름에서 _asr 접미사 제거 → 학습 출력 디렉토리 추정
    guess = run_name.replace("_asr", "")
    f = Path(hist_root) / guess / "history.json"
    if not f.exists():
        return {}
    try:
        hist = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(hist, list) and hist:
            last = hist[-1]
            return {k: v for k, v in last.items() if k.startswith("diag_")}
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def group_seeds(rows: Dict[str, dict]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for name in rows:
        base = SEED_RE.sub("", name)
        groups.setdefault(base, []).append(name)
    return groups


def build_table(rows: Dict[str, dict], hist_root: Optional[str]) -> List[dict]:
    groups = group_seeds(rows)
    table: List[dict] = []
    for base, names in sorted(groups.items()):
        per_attack: Dict[str, List[float]] = {a: [] for a in ATTACKS}
        means, fids, clips = [], [], []
        for n in names:
            vals, mean = extract_asr(rows[n])
            for a in ATTACKS:
                if vals[a] is not None:
                    per_attack[a].append(vals[a])
            if mean is not None:
                means.append(mean)
            fid, clip = extract_quality(rows[n])
            if fid is not None:
                fids.append(fid)
            if clip is not None:
                clips.append(clip)
        entry = {"name": base, "n_seeds": len(names)}
        for a in ATTACKS:
            xs = per_attack[a]
            entry[a] = sum(xs) / len(xs) if xs else None
            entry[a + "_std"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        entry["mean_asr"] = sum(means) / len(means) if means else None
        entry["mean_asr_std"] = statistics.pstdev(means) if len(means) > 1 else 0.0
        entry["fid"] = sum(fids) / len(fids) if fids else None
        entry["clip"] = sum(clips) / len(clips) if clips else None
        entry["diag"] = load_final_diag(hist_root, names[0])
        table.append(entry)
    table.sort(key=lambda e: (e["mean_asr"] if e["mean_asr"] is not None else 1e9))
    return table


def to_markdown(table: List[dict]) -> str:
    head = "| Run | seeds | " + " | ".join(ATTACKS) + " | **mean ASR** | FID | CLIP | resid | gcos |"
    sep = "|" + "---|" * (len(ATTACKS) + 7)
    lines = [head, sep]
    for e in table:
        cells = [e["name"], str(e["n_seeds"])]
        for a in ATTACKS:
            s = _fmt(e[a])
            if e[a] is not None and e.get(a + "_std", 0) > 0.001:
                s += f"±{e[a+'_std']*100:.0f}"
            cells.append(s)
        m = _fmt(e["mean_asr"])
        if e["mean_asr"] is not None and e["mean_asr_std"] > 0.001:
            m += f"±{e['mean_asr_std']*100:.0f}"
        cells.append(f"**{m}**")
        cells.append(_fmt(e["fid"], pct=False) if e["fid"] is not None else "-")
        cells.append(_fmt(e["clip"], pct=False) if e["clip"] is not None else "-")
        diag = e.get("diag", {})
        cells.append(f"{diag['diag_residual_var']:.2f}" if "diag_residual_var" in diag else "-")
        cells.append(f"{diag['diag_grad_cosine']:.2f}" if "diag_grad_cosine" in diag else "-")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_csv(table: List[dict], path: str):
    fields = (["name", "n_seeds"] + ATTACKS + ["mean_asr", "mean_asr_std", "fid", "clip"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for e in table:
            w.writerow([e.get(k, "") if e.get(k) is not None else "" for k in fields])


def main():
    ap = argparse.ArgumentParser(description="LSSE 결과 집계")
    ap.add_argument("--eval_root", default="outputs/eval")
    ap.add_argument("--hist_root", default="outputs",
                    help="history.json 위치(진단 컬럼). 비우려면 'none'")
    ap.add_argument("--out", default="outputs/summary.md")
    args = ap.parse_args()

    hist_root = None if args.hist_root.lower() == "none" else args.hist_root
    rows = load_eval_dir(args.eval_root)
    if not rows:
        print(f"[aggregate] {args.eval_root} 에 eval_results.json 없음")
        return
    table = build_table(rows, hist_root)
    md = to_markdown(table)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# LSSE 실험 요약\n\nASR↓ 좋음 / FID↓ CLIP↑ 좋음 / resid↓ gcos>0 좋음\n\n" + md + "\n",
                   encoding="utf-8")
    csv_path = str(out.with_suffix(".csv"))
    to_csv(table, csv_path)
    print(md)
    print(f"\n[aggregate] -> {out}  &  {csv_path}")


if __name__ == "__main__":
    main()
