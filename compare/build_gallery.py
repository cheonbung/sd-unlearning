"""Build a static side-by-side gallery for qualitative model comparison.

The generated HTML references existing eval images by relative path. It does
not copy or embed images, so it stays small and can be opened directly.
"""

from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_HTML = REPO_ROOT / "compare" / "comparison_gallery.html"


@dataclass(frozen=True)
class Attack:
    key: str
    label: str
    prompt_file: str
    folder_variants: tuple[str, ...]


@dataclass(frozen=True)
class Model:
    key: str
    label: str
    eval_dir: str
    note: str


ATTACKS = [
    Attack("I2P", "I2P", "lsse/data/eval/i2p_nudity.txt", ("i2p",)),
    Attack(
        "Ring-A-Bell",
        "Ring-A-Bell",
        "lsse/data/eval/ring_a_bell_nudity.txt",
        ("ring_a_bell",),
    ),
    Attack(
        "Ring-A-Bell(Re)",
        "Ring-A-Bell(Re)",
        "lsse/data/eval/ring_a_bell_re_nudity.txt",
        ("ring_a_bell_re", "ring_a_bellre"),
    ),
    Attack("P4D", "P4D", "lsse/data/eval/p4d_nudity.txt", ("p4d",)),
    Attack(
        "UnlearnDiffAtk",
        "UnlearnDiffAtk",
        "lsse/data/eval/unlearnDiffAtk_nudity.txt",
        ("unlearnDiffAtk", "unlearndiffatk"),
    ),
]


MODELS = [
    Model(
        "raw_sd",
        "Raw SD",
        "lsse/outputs/eval/xharness_rawsd",
        "Unlearned SD v1-4",
    ),
    Model(
        "fcf_p",
        "FCF-P",
        "lsse/outputs/eval/xharness_fcf_p",
        "FCF projection baseline",
    ),
    Model(
        "fcf_e",
        "FCF-E",
        "lsse/outputs/eval/xharness_fcf_e",
        "FCF empirical baseline",
    ),
    Model(
        "vanilla_lsse",
        "Vanilla LSSE",
        "lsse/outputs/eval/baseline_seed42",
        "CNP + CSR + CLM",
    ),
    Model(
        "lsse_plu",
        "LSSE+PLU",
        "lsse/outputs/eval/plu_seed42",
        "Progressive layer unlocking",
    ),
    Model(
        "lsse_plu_w2",
        "LSSE+PLU+W2",
        "lsse/outputs/eval/stack_plu_w2_seed42",
        "PLU + margin CNP",
    ),
    Model(
        "sph_ot",
        "Spherical+OT",
        "lsse/outputs/eval/xharness_sph_ot",
        "N5 spherical + N6 OT noise",
    ),
    Model(
        "dace",
        "DACE",
        "lsse/outputs/eval/xharness_dace",
        "Dynamic concept-axis erasure",
    ),
    Model(
        "dace_plu",
        "DACE+PLU",
        "lsse/outputs/eval/xharness_dace_plu",
        "DACE with PLU",
    ),
    Model(
        "odace_v2",
        "ODACE earlier",
        "odace/outputs/eval/odace",
        "Earlier UNet output-grounded run",
    ),
    Model(
        "odace_v3",
        "ODACE v3",
        "odace/outputs/eval/odace_v3",
        "Full cross-attention UNet edit",
    ),
]


def rel_from_html(path: Path) -> str:
    return os.path.relpath(path, OUT_HTML.parent).replace(os.sep, "/")


def read_prompts(rel_path: str) -> list[str]:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return []
    prompts: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if text and not text.startswith("#"):
            prompts.append(text)
    return prompts


def load_asr(model: Model) -> tuple[dict[str, float | None], float | None]:
    path = REPO_ROOT / model.eval_dir / "eval_results.json"
    if not path.exists():
        return {}, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}, None
    raw = data.get("asr", data)
    vals: dict[str, float | None] = {}
    for attack in ATTACKS:
        item = raw.get(attack.key)
        if isinstance(item, dict):
            vals[attack.key] = item.get("asr")
        elif isinstance(item, (int, float)):
            vals[attack.key] = float(item)
        else:
            vals[attack.key] = None
    present = [v for v in vals.values() if isinstance(v, (int, float))]
    mean = sum(present) / len(present) if present else None
    return vals, mean


def image_base_dir(model: Model, attack: Attack) -> Path | None:
    root = REPO_ROOT / model.eval_dir / "images" / "fcf_nudity"
    for folder in attack.folder_variants:
        cand = root / folder
        if cand.exists():
            return cand
    return None


def image_path(model: Model, attack: Attack, index: int) -> Path | None:
    base = image_base_dir(model, attack)
    if base is None:
        return None
    cand = base / f"{index:04d}_00.png"
    return cand if cand.exists() else None


def count_images(model: Model, attack: Attack) -> int:
    base = image_base_dir(model, attack)
    if base is None:
        return 0
    return len(list(base.glob("*.png")))


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def css() -> str:
    return """
:root {
  --bg: #111315;
  --panel: #191c20;
  --panel-2: #22262b;
  --text: #eff2f4;
  --muted: #a8b0b8;
  --line: #343a42;
  --accent: #5ec6a8;
  --warn: #e7b456;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
header {
  position: sticky;
  top: 0;
  z-index: 20;
  background: rgba(17, 19, 21, 0.96);
  border-bottom: 1px solid var(--line);
  padding: 14px 18px;
}
h1 {
  margin: 0 0 10px;
  font-size: 20px;
  font-weight: 700;
  letter-spacing: 0;
}
.toolbar, .model-toggle {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
button, input, label.toggle {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--text);
  border-radius: 6px;
  padding: 8px 10px;
  font: inherit;
}
button {
  cursor: pointer;
}
button.active {
  border-color: var(--accent);
  background: #17352f;
}
label.toggle {
  display: inline-flex;
  gap: 7px;
  align-items: center;
}
.model-toggle {
  margin-top: 10px;
}
.summary {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 8px;
  padding: 14px 18px 4px;
}
.summary-card {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 6px;
  padding: 10px;
  min-height: 78px;
}
.summary-card strong {
  display: block;
  font-size: 14px;
}
.summary-card span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-top: 2px;
}
.attack-section {
  padding: 16px 18px 34px;
}
.attack-title {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin: 6px 0 12px;
}
.attack-title h2 {
  margin: 0;
  font-size: 18px;
}
.attack-title span {
  color: var(--muted);
}
.table-wrap {
  overflow-x: auto;
  border: 1px solid var(--line);
  border-radius: 8px;
}
.grid {
  display: grid;
  grid-template-columns: 280px repeat(var(--model-count), minmax(164px, 1fr));
  min-width: calc(280px + var(--model-count) * 164px);
}
.cell {
  border-right: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  padding: 8px;
}
.cell.header {
  background: var(--panel-2);
  font-weight: 700;
  min-height: 54px;
}
.prompt {
  color: var(--muted);
  font-size: 12px;
  line-height: 1.45;
}
.prompt b {
  color: var(--text);
  font-size: 13px;
}
.image-cell {
  min-height: 188px;
}
.image-cell a {
  display: block;
}
img {
  width: 100%;
  aspect-ratio: 1 / 1;
  object-fit: cover;
  border-radius: 6px;
  background: #0b0c0d;
  transition: filter 120ms ease;
}
body.blur-on img {
  filter: blur(18px) saturate(0.5);
}
.meta {
  color: var(--muted);
  font-size: 11px;
  margin-top: 6px;
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.missing {
  display: grid;
  place-items: center;
  height: 156px;
  color: var(--warn);
  border: 1px dashed var(--line);
  border-radius: 6px;
  font-size: 12px;
}
.hidden { display: none !important; }
.model-off { visibility: hidden; pointer-events: none; }
@media (max-width: 760px) {
  .grid {
    grid-template-columns: 220px repeat(var(--model-count), minmax(150px, 1fr));
    min-width: calc(220px + var(--model-count) * 150px);
  }
}
"""


def js() -> str:
    return """
const body = document.body;
const buttons = [...document.querySelectorAll('[data-attack-button]')];
const sections = [...document.querySelectorAll('[data-attack-section]')];
const promptInput = document.getElementById('promptFilter');
const blurToggle = document.getElementById('blurToggle');
const missingToggle = document.getElementById('missingToggle');
const modelChecks = [...document.querySelectorAll('[data-model-toggle]')];

function setAttack(key) {
  buttons.forEach(btn => btn.classList.toggle('active', btn.dataset.attackButton === key));
  sections.forEach(sec => sec.classList.toggle('hidden', key !== 'all' && sec.dataset.attackSection !== key));
}

function applyFilters() {
  const q = promptInput.value.trim();
  const showMissing = missingToggle.checked;
  document.querySelectorAll('[data-prompt-row]').forEach(row => {
    const matchPrompt = !q || row.dataset.promptIndex.includes(q);
    const hasMissing = row.dataset.hasMissing === '1';
    row.classList.toggle('hidden', !matchPrompt || (!showMissing && hasMissing));
  });
  modelChecks.forEach(check => {
    const key = check.dataset.modelToggle;
    document.querySelectorAll(`[data-model="${key}"]`).forEach(cell => {
      cell.classList.toggle('model-off', !check.checked);
    });
  });
}

buttons.forEach(btn => btn.addEventListener('click', () => setAttack(btn.dataset.attackButton)));
promptInput.addEventListener('input', applyFilters);
missingToggle.addEventListener('change', applyFilters);
modelChecks.forEach(check => check.addEventListener('change', applyFilters));
blurToggle.addEventListener('change', () => body.classList.toggle('blur-on', blurToggle.checked));
setAttack('all');
applyFilters();
"""


def render_summary(model_stats: dict[str, tuple[dict[str, float | None], float | None]]) -> str:
    cards = []
    for model in MODELS:
        asr, mean = model_stats[model.key]
        cards.append(
            '<div class="summary-card">'
            f"<strong>{html.escape(model.label)}</strong>"
            f"<span>mean ASR {pct(mean)}</span>"
            f"<span>{html.escape(model.note)}</span>"
            "</div>"
        )
    return "\n".join(cards)


def render_model_toggles() -> str:
    toggles = []
    for model in MODELS:
        toggles.append(
            '<label class="toggle">'
            f'<input type="checkbox" checked data-model-toggle="{html.escape(model.key)}">'
            f"{html.escape(model.label)}"
            "</label>"
        )
    return "\n".join(toggles)


def row_has_missing(paths: Iterable[Path | None]) -> bool:
    return any(path is None for path in paths)


def render_attack(attack: Attack, model_stats: dict[str, tuple[dict[str, float | None], float | None]]) -> str:
    prompts = read_prompts(attack.prompt_file)
    counts = [count_images(model, attack) for model in MODELS]
    n_rows = min(max(counts), len(prompts) if prompts else 50, 50)
    if n_rows == 0:
        n_rows = 50

    headers = ['<div class="cell header">Prompt</div>']
    for model in MODELS:
        asr, mean = model_stats[model.key]
        headers.append(
            f'<div class="cell header" data-model="{html.escape(model.key)}">'
            f"{html.escape(model.label)}"
            f'<div class="meta"><span>{html.escape(attack.label)} {pct(asr.get(attack.key))}</span>'
            f"<span>mean {pct(mean)}</span></div>"
            "</div>"
        )

    rows: list[str] = []
    for idx in range(n_rows):
        paths = [image_path(model, attack, idx) for model in MODELS]
        missing = "1" if row_has_missing(paths) else "0"
        prompt = prompts[idx] if idx < len(prompts) else ""
        row_attrs = (
            f'data-prompt-row data-prompt-index="{idx:04d}" '
            f'data-has-missing="{missing}"'
        )
        rows.append(
            f'<div class="cell prompt" {row_attrs}>'
            f"<b>#{idx:04d}</b><br>{html.escape(prompt)}"
            "</div>"
        )
        for model, path in zip(MODELS, paths):
            if path is None:
                body = '<div class="missing">missing</div>'
            else:
                rel = rel_from_html(path)
                body = (
                    f'<a href="{html.escape(rel)}" target="_blank" rel="noopener">'
                    f'<img loading="lazy" src="{html.escape(rel)}" alt="{html.escape(model.label)} {idx:04d}">'
                    "</a>"
                )
            rows.append(
                f'<div class="cell image-cell" {row_attrs} data-model="{html.escape(model.key)}">'
                f"{body}"
                f'<div class="meta"><span>{html.escape(model.label)}</span><span>{idx:04d}</span></div>'
                "</div>"
            )

    return (
        f'<section class="attack-section" data-attack-section="{html.escape(attack.key)}">'
        '<div class="attack-title">'
        f"<h2>{html.escape(attack.label)}</h2>"
        f"<span>{n_rows} prompts</span>"
        "</div>"
        '<div class="table-wrap">'
        f'<div class="grid" style="--model-count:{len(MODELS)}">'
        + "\n".join(headers + rows)
        + "</div></div></section>"
    )


def build_html() -> str:
    model_stats = {model.key: load_asr(model) for model in MODELS}
    attack_buttons = ['<button class="active" data-attack-button="all">All</button>']
    attack_buttons.extend(
        f'<button data-attack-button="{html.escape(attack.key)}">{html.escape(attack.label)}</button>'
        for attack in ATTACKS
    )

    sections = "\n".join(render_attack(attack, model_stats) for attack in ATTACKS)

    return (
        "<!doctype html><html lang=\"ko\"><head><meta charset=\"utf-8\">"
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>SD Unlearning Qualitative Gallery</title>"
        f"<style>{css()}</style></head><body class=\"blur-on\">"
        "<header>"
        "<h1>SD Unlearning Qualitative Gallery</h1>"
        '<div style="margin:0 0 10px;padding:9px 12px;border:1px solid var(--warn);'
        'border-radius:6px;background:#2a2415;color:var(--text);font-size:13px;line-height:1.5">'
        '<strong style="color:var(--warn)">⚠️ FCF-P/E 정정 안내</strong> — '
        '아래 <b>FCF-P(52.8%)·FCF-E(61.2%)</b> 열·이미지는 '
        '<b>우리 <code>fcf/</code> 재구현</b>(불충실: 단어리스트 vs '
        '문장삼중쌍, <code>/(1−η)</code> 정규화 누락)의 결과다. '
        '<b>저자 공식 코드+데이터로 재학습</b>하면 FCF-P는 '
        '논문정렬 full-set 4-label ASR <b>3.7 ≈ 논문 3.43</b>(8-label 16.9)로 '
        '재현되며 ESD-u(21.6)보다 낮다. 자세한 정정표는 '
        '<code>comparison_all_methods.md</code> §③-정정 참조.</div>'
        '<div class="toolbar">'
        + "\n".join(attack_buttons)
        + '<input id="promptFilter" type="search" placeholder="prompt index, e.g. 0007">'
        + '<label class="toggle"><input id="blurToggle" type="checkbox" checked> blur images</label>'
        + '<label class="toggle"><input id="missingToggle" type="checkbox" checked> show rows with missing images</label>'
        + "</div>"
        f'<div class="model-toggle">{render_model_toggles()}</div>'
        "</header>"
        f'<main><section class="summary">{render_summary(model_stats)}</section>{sections}</main>'
        f"<script>{js()}</script></body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build qualitative comparison gallery")
    parser.add_argument("--out", default=str(OUT_HTML), help="Output HTML path")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
