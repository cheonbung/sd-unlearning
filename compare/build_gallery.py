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
    images_subdir: str = "images/fcf_nudity"
    # Explicit per-attack ASR (fraction 0-1). When set, used instead of reading
    # eval_results.json -- needed for the official FCF checkpoints whose ASR comes
    # from the FCF-protocol re-score (compare/fcf_rescore.json), not a per-eval JSON.
    asr: "dict[str, float] | None" = None


ATTACKS = [
    Attack("I2P", "I2P", "models/lsse/data/eval/i2p_nudity.txt", ("i2p",)),
    Attack(
        "Ring-A-Bell",
        "Ring-A-Bell",
        "models/lsse/data/eval/ring_a_bell_nudity.txt",
        ("ring_a_bell",),
    ),
    Attack(
        "Ring-A-Bell(Re)",
        "Ring-A-Bell(Re)",
        "models/lsse/data/eval/ring_a_bell_re_nudity.txt",
        ("ring_a_bell_re", "ring_a_bellre"),
    ),
    Attack("P4D", "P4D", "models/lsse/data/eval/p4d_nudity.txt", ("p4d",)),
    Attack(
        "UnlearnDiffAtk",
        "UnlearnDiffAtk",
        "models/lsse/data/eval/unlearnDiffAtk_nudity.txt",
        ("unlearnDiffAtk", "unlearndiffatk"),
    ),
]


MODELS = [
    Model(
        "raw_sd",
        "Raw SD",
        "models/lsse/outputs/eval/xharness_rawsd",
        "Unlearned SD v1-4",
    ),
    Model(
        "fcf_p",
        "FCF-P",
        "eval/outputs/fcf_p_official",
        "FCF-P (official authors' code reproduction)",
        images_subdir="attacks",
        asr={"I2P": 0.10, "Ring-A-Bell": 0.24, "Ring-A-Bell(Re)": 0.20,
             "P4D": 0.10, "UnlearnDiffAtk": 0.22},
    ),
    Model(
        "fcf_e",
        "FCF-E",
        "eval/outputs/fcf_e_official",
        "FCF-E (official authors' code reproduction)",
        images_subdir="attacks",
        asr={"I2P": 0.16, "Ring-A-Bell": 0.28, "Ring-A-Bell(Re)": 0.36,
             "P4D": 0.14, "UnlearnDiffAtk": 0.46},
    ),
    Model(
        "vanilla_lsse",
        "Vanilla LSSE",
        "models/lsse/outputs/eval/baseline_seed42",
        "CNP + CSR + CLM",
    ),
    Model(
        "lsse_plu",
        "LSSE+PLU",
        "models/lsse/outputs/eval/plu_seed42",
        "Progressive layer unlocking",
    ),
    Model(
        "lsse_plu_w2",
        "LSSE+PLU+W2",
        "models/lsse/outputs/eval/stack_plu_w2_seed42",
        "PLU + margin CNP",
    ),
    Model(
        "sph_ot",
        "Spherical+OT",
        "models/lsse/outputs/eval/xharness_sph_ot",
        "N5 spherical + N6 OT noise",
    ),
    Model(
        "dace",
        "DACE",
        "models/lsse/outputs/eval/xharness_dace",
        "Dynamic concept-axis erasure",
    ),
    Model(
        "dace_plu",
        "DACE+PLU",
        "models/lsse/outputs/eval/xharness_dace_plu",
        "DACE with PLU",
    ),
    Model(
        "odace_v2",
        "ODACE earlier",
        "models/odace/outputs/eval/odace",
        "Earlier UNet output-grounded run",
    ),
    Model(
        "odace_v3",
        "ODACE v3",
        "models/odace/outputs/eval/odace_v3",
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
    if model.asr is not None:
        vals = {a.key: model.asr.get(a.key) for a in ATTACKS}
        present = [v for v in vals.values() if isinstance(v, (int, float))]
        mean = sum(present) / len(present) if present else None
        return vals, mean
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
    root = REPO_ROOT / model.eval_dir / model.images_subdir
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


# 4-label (FCF paper rule: 4 fully-exposed labels) ASR per gallery model, read from the re-score
# JSON so the gallery shows BOTH the harness 8-label and FCF's 4-label number on the same images.
RESCORE_KEY = {
    "raw_sd": "raw_v14", "fcf_p": "fcf_p_official", "fcf_e": "fcf_e_official",
    "vanilla_lsse": "vanilla_lsse", "lsse_plu": "lsse_plu", "lsse_plu_w2": "lsse_plu_w2",
    "sph_ot": "sph_ot", "dace": "dace", "dace_plu": "dace_plu",
    "odace_v2": "odace_v2", "odace_v3": "odace_v3",
}
RESCORE_ATTACK = {
    "I2P": "original_i2p", "Ring-A-Bell": "ring_a_bell", "Ring-A-Bell(Re)": "ring_a_bell_re",
    "P4D": "p4d", "UnlearnDiffAtk": "unlearndiff",
}


def load_asr4() -> "dict[str, tuple[dict[str, float | None], float | None]]":
    """gallery model key -> (per-attack 4-label fraction dict, mean fraction) from fcf_rescore.json."""
    path = REPO_ROOT / "compare" / "fcf_rescore.json"
    if not path.exists():
        return {}
    try:
        methods = json.loads(path.read_text(encoding="utf-8")).get("methods", {})
    except json.JSONDecodeError:
        return {}
    out: dict[str, tuple[dict[str, float | None], float | None]] = {}
    for gkey, rkey in RESCORE_KEY.items():
        rec = methods.get(rkey)
        if not rec:
            continue
        per: dict[str, float | None] = {}
        for akey, rakey in RESCORE_ATTACK.items():
            item = rec.get("attacks", {}).get(rakey)
            v = item.get("asr_fcf") if isinstance(item, dict) else None
            per[akey] = v / 100.0 if isinstance(v, (int, float)) else None
        m = rec.get("asr_fcf_mean")
        out[gkey] = (per, m / 100.0 if isinstance(m, (int, float)) else None)
    return out


ASR4 = load_asr4()


def render_summary(model_stats: dict[str, tuple[dict[str, float | None], float | None]]) -> str:
    cards = []
    for model in MODELS:
        asr, mean = model_stats[model.key]
        _per4, mean4 = ASR4.get(model.key, ({}, None))
        mean4_txt = f" · 4-lab {pct(mean4)}" if mean4 is not None else ""
        cards.append(
            '<div class="summary-card">'
            f"<strong>{html.escape(model.label)}</strong>"
            f"<span>mean ASR {pct(mean)}{mean4_txt}</span>"
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
        per4, mean4 = ASR4.get(model.key, ({}, None))
        a4 = per4.get(attack.key)
        a4_txt = f" · 4-lab {pct(a4)}" if a4 is not None else ""
        mean4_txt = f" · 4-lab {pct(mean4)}" if mean4 is not None else ""
        headers.append(
            f'<div class="cell header" data-model="{html.escape(model.key)}">'
            f"{html.escape(model.label)}"
            f'<div class="meta"><span>{html.escape(attack.label)} {pct(asr.get(attack.key))}{a4_txt}</span>'
            f"<span>mean {pct(mean)}{mean4_txt}</span></div>"
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
        '<div style="margin:0 0 10px;color:#a8b0b8;font-size:12px">ASR 표기: '
        '<b>8-lab</b> = 본 harness 8라벨(NudeNet score&gt;0.3) · '
        '<b>4-lab</b> = FCF 논문의 완전노출 4라벨로 동일 이미지 재채점(부분집합 → 항상 ≤ 8-lab)</div>'
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
