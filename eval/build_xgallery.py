"""Build a self-contained HTML gallery for the cross-model attack comparison.

Columns = models (raw v1.4/v1.5, safe_neg, ODACE v3/v1.5); sections = the 5 nudity attacks.
Same prompt index = same prompt across models. Each column header shows the model's measured
ASR (NudeNet v3) from its metrics.json, so the "low ASR but off-target" story is visible by
eye: scan an attack row and compare what each model renders. Locality/specificity is reported
separately by the standard COCO protocol (eval_coco.py), not here. Open
eval/comparison_xmodel.html in a browser.

Run:  python eval/build_xgallery.py
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import xeval  # noqa: E402  (REGISTRY, ATTACKS, find_sub, read_prompts)

OUT_HTML = HERE / "comparison_xmodel.html"
MODELS = ["raw_v14", "raw_v15", "safe_neg", "sd21base",
          "esd_u", "sld_medium", "sld_strong", "sld_max", "safeclip",
          "odace_v3", "odace_v15"]
LABELS = {"raw_v14": "raw v1.4", "raw_v15": "raw v1.5", "safe_neg": "safe_neg (v1.5+neg)",
          "sd21base": "SD2.1-base (NSFW-filter)",
          "esd_u": "ESD-u (v1.4)", "sld_medium": "SLD-Medium", "sld_strong": "SLD-Strong",
          "sld_max": "SLD-Max", "safeclip": "Safe-CLIP",
          "odace_v3": "ODACE v3 (v1.4)", "odace_v15": "ODACE v1.5"}


def load_metrics(label):
    f = HERE / "outputs" / label / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else {}


def attack_dir(label, variants):
    spec = xeval.REGISTRY[label]
    root = (REPO / spec["attack_dir"]) if "attack_dir" in spec else (HERE / "outputs" / label / "attacks")
    return xeval.find_sub(root, variants)


def rel(path: Path):
    return os.path.relpath(path, HERE).replace(os.sep, "/")


def asr_color(p):
    if p is None: return "#888"
    if p <= 10: return "#1a9850"
    if p <= 25: return "#66bd63"
    if p <= 45: return "#fee08b"
    if p <= 60: return "#fc8d59"
    return "#d73027"


CSS = """
*{box-sizing:border-box}body{margin:0;background:#0f1115;color:#e6e6e6;font:14px/1.4 system-ui,Segoe UI,sans-serif}
header{position:sticky;top:0;z-index:30;background:#161922;padding:12px 16px;border-bottom:1px solid #2a2f3a;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
header h1{font-size:16px;margin:0}.nav a{color:#9cd;text-decoration:none;margin-right:10px;font-size:13px}
button{background:#2a3142;color:#e6e6e6;border:1px solid #3a4252;border-radius:6px;padding:6px 12px;cursor:pointer}
.legend{font-size:12px;color:#9aa}.legend b{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:middle}
section{padding:8px 16px 28px}section h2{position:sticky;top:49px;z-index:20;background:#0f1115;margin:0 -16px 8px;padding:10px 16px;border-bottom:1px solid #2a2f3a;font-size:15px}
table{border-collapse:separate;border-spacing:0;width:max-content}
thead th{position:sticky;top:92px;z-index:10;background:#1b1f2a;padding:6px 8px;border-bottom:2px solid #2a2f3a;text-align:center;font-size:12px;min-width:152px}
thead th.pc{left:0;z-index:12;text-align:left;min-width:240px;max-width:240px}
td{padding:3px;border-bottom:1px solid #1c2029;vertical-align:top;text-align:center}
td.pc{position:sticky;left:0;z-index:5;background:#12151c;text-align:left;min-width:240px;max-width:240px;font-size:12px;color:#c7c7c7;padding:6px 8px}
td.pc .idx{color:#7a8;font-weight:700;margin-right:6px}
.cell img{width:148px;height:148px;object-fit:cover;border-radius:5px;display:block;filter:blur(15px);transition:filter .12s;cursor:pointer;background:#222}
.cell img.revealed{filter:none}body.revealall .cell img{filter:none}
.miss{width:148px;height:148px;display:flex;align-items:center;justify-content:center;color:#556;font-size:11px;border:1px dashed #2a2f3a;border-radius:5px}
.masr{font-size:11px}
"""
JS = """
function tb(){document.body.classList.toggle('revealall');var b=document.getElementById('bb');
b.textContent=document.body.classList.contains('revealall')?'🙈 Blur all':'👁 Reveal all';}
document.addEventListener('click',function(e){if(e.target.tagName==='IMG'&&e.target.classList.contains('thumb')){
if(e.shiftKey){window.open(e.target.src,'_blank');}else{e.target.classList.toggle('revealed');}}});
"""

metrics = {m: load_metrics(m) for m in MODELS}
parts = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'>",
         "<title>교차모델 공격 비교</title><style>", CSS, "</style></head><body>"]
SECTIONS = [(k, pf, v) for k, pf, v in xeval.ATTACKS]
nav = " ".join(f"<a href='#{k.replace('(','').replace(')','')}'>{html.escape(k)}</a>" for k, _, _ in SECTIONS)
parts.append("<header><h1>교차모델 · 공격 비교</h1>")
parts.append("<button id='bb' onclick='tb()'>👁 Reveal all</button>")
parts.append(f"<span class='nav'>{nav}</span>")
parts.append("<span class='legend'>열=모델(왼쪽 raw→오른쪽 ODACE) · 공격행에서 각 모델이 무엇을 "
             "생성하는지 비교(ODACE는 프롬프트를 벗어남) · 일반-충실도는 COCO(eval_coco.py)에서 별도 측정 · "
             "클릭=개별보기 Shift+클릭=원본 · "
             "<b style='background:#1a9850'></b>ASR↓ <b style='background:#d73027'></b>ASR↑</span></header>")

for key, pfile, variants in SECTIONS:
    anchor = key.replace('(', '').replace(')', '')
    prompts = xeval.read_prompts(REPO / "models/lsse/data/eval" / pfile)
    dirs = {m: attack_dir(m, variants) for m in MODELS}
    parts.append(f"<section id='{anchor}'><h2>{html.escape(key)} "
                 f"<span style='font-weight:400;color:#889;font-size:12px'>"
                 f"({pfile})</span></h2>")
    parts.append("<table><thead><tr><th class='pc'>프롬프트</th>")
    for m in MODELS:
        mm = metrics.get(m, {})
        a = (mm.get("asr") or {}).get(key)
        col = asr_color(a)
        extra = f"<span style='color:{col}'>ASR {a if a is not None else '-'}</span>"
        parts.append(f"<th><div style='font-weight:700'>{html.escape(LABELS[m])}</div>"
                     f"<div class='masr' style='color:#bbb'>{extra}</div></th>")
    parts.append("</tr></thead><tbody>")
    for i in range(len(prompts)):
        ptxt = prompts[i]
        ptxt = (ptxt[:150] + "…") if len(ptxt) > 150 else ptxt
        parts.append(f"<tr><td class='pc'><span class='idx'>#{i:02d}</span>{html.escape(ptxt)}</td>")
        for m in MODELS:
            d = dirs[m]
            img = (d / f"{i:04d}_00.png") if d else None
            if img and img.exists():
                src = rel(img)
                parts.append(f"<td class='cell'><img class='thumb' loading='lazy' src='{src}' "
                             f"alt='{html.escape(LABELS[m])} #{i}'></td>")
            else:
                parts.append("<td class='cell'><div class='miss'>없음</div></td>")
        parts.append("</tr>")
    parts.append("</tbody></table></section>")

parts.append("<script>" + JS + "</script></body></html>")
OUT_HTML.write_text("".join(parts), encoding="utf-8")
print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size // 1024} KB)")
for m in MODELS:
    mm = metrics.get(m, {})
    print(f"  {LABELS[m]:20s} ASR={mm.get('asr_mean')}")
