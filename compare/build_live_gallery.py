"""Live full-set gallery + quantitative table over eval/outputs/<model>_fs/.

Two things in one page:
  1. Quantitative summary TABLES (models as rows). The FULL-SET table shows
     nudity ASR per attack plus two means - 8-lab (our rule) and 4-lab (FCF
     exposed-only), both score>0.3 - read from fullset_all.json / fullset_eval.json,
     with COCO-FID/CLIP + Violence(Q16). A second LEGACY table shows the earlier
     50-prompt re-score from eval/outputs/<key>/metrics.json. All values are read
     at BUILD time and embedded (works over file://). Missing -> "—".
  2. A live image GALLERY (models as columns) over the FULL-SET images that
     eval_fullset_all.py writes to eval/outputs/<key>_fs/. Every cell points at
     the FINAL path even if the file does not exist yet; a JS timer retries
     not-yet-loaded images (cache-busted) every 25s, so the gallery fills in
     LIVE as the background job generates each image.

Condition toggles (group / base / modification) filter BOTH the table rows and
the gallery columns at once. Reference (Raw SD) models can be pinned so they
stay visible for comparison regardless of the active filter.

Run:  python compare/build_live_gallery.py [--rows 80] [--out PATH]
      --rows 0  -> all prompts per attack (large page)
"""
from __future__ import annotations
import argparse
import html
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "compare" / "comparison_gallery_live.html"
FS_ROOT = REPO / "eval" / "outputs"
PROMPT_DIR = REPO / "models" / "fcf" / "data" / "eval"
VIOLENCE_JSON = REPO / "models" / "fcf" / "violence_q16.json"
STYLE_JSON = REPO / "models" / "fcf" / "style_vangogh.json"   # Van Gogh style locality (img2raw)
VIOLENCE50_JSON = REPO / "models" / "fcf" / "violence_q16_smoke.json"  # 50-prompt violence subset
TRAINCOST_JSON = REPO / "models" / "fcf" / "train_cost.json"   # env-aware training cost (GPU-h)
FULLSET_ALL = REPO / "models" / "fcf" / "fullset_all.json"     # 17 Table-A methods (frozen full-set)
FULLSET_EVAL = REPO / "models" / "fcf" / "fullset_eval.json"   # raw_v14 / fcf_p / fcf_e

# (display label, key, group, base, modification site)
#   key       -> images at eval/outputs/<key>_fs/, metrics at eval/outputs/<key>/
#   group     -> ref | baseline | core | novel
#   base      -> 1.4 | 1.5 | 2.1   (SD backbone)
#   mod       -> none | guidance | clip | text | unet   (what the method edits)
MODELS = [
    ("Raw SD v1.4",            "raw_v14",        "ref",      "1.4", "none"),
    ("Raw SD v1.5",            "raw_v15",        "ref",      "1.5", "none"),
    ("SD2.1-base",             "sd21base",       "ref",      "2.1", "none"),
    ("SLD-Medium",             "sld_medium",     "baseline", "1.4", "guidance"),
    ("SLD-Strong",             "sld_strong",     "baseline", "1.4", "guidance"),
    ("SLD-Max",                "sld_max",        "baseline", "1.4", "guidance"),
    ("Safe-CLIP",              "safeclip",       "baseline", "1.4", "clip"),
    ("Safe-neg",               "safe_neg",       "baseline", "1.4", "guidance"),
    ("ESD-u",                  "esd_u",          "baseline", "1.4", "unet"),
    ("FCF-P",                  "fcf_p_official", "core",     "1.4", "text"),
    ("FCF-E",                  "fcf_e_official", "core",     "1.4", "text"),
    ("LSSE",                   "vanilla_lsse",   "core",     "1.4", "text"),
    ("LSSE+PLU",               "lsse_plu",       "core",     "1.4", "text"),
    ("LSSE+PLU+W2",            "lsse_plu_w2",    "core",     "1.4", "text"),
    ("SLERP-OT",               "sph_ot",         "novel",    "1.4", "text"),
    ("DACE",                   "dace_v2",        "core",     "1.4", "text"),
    ("DACE+PLU",               "dace_plu",       "core",     "1.4", "text"),
    ("ODACE (SD v1.4)",        "odace_v3",       "novel",    "1.4", "unet"),
    ("ODACE (SD v1.5)",        "odace_v15",      "novel",    "1.5", "unet"),
    ("ODACE (SD v1.4, early)", "odace_v2",       "novel",    "1.4", "unet"),
    ("LSSE-MC (n+v+vg)",       "lsse_mc_nvg",     "novel",    "1.4", "text"),
]
# (label, _fs subdir, prompt file)
ATTACKS = [
    ("I2P", "i2p", "i2p_nudity.txt"),
    ("Ring-A-Bell", "ring_a_bell", "ring_a_bell_nudity.txt"),
    ("Ring-A-Bell(Re)", "ring_a_bell_re", "ring_a_bell_re_nudity.txt"),
    ("P4D", "p4d", "p4d_nudity.txt"),
    ("UnlearnDiffAtk", "unlearndiffatk", "unlearnDiffAtk_nudity.txt"),
]
# (dimension key, display label, [(value, chip label), ...])
FILTERS = [
    ("group", "Group", [("ref", "Reference"), ("baseline", "Baseline"),
                        ("core", "Core (FCF/LSSE/DACE)"), ("novel", "Novel (ODACE/SLERP-OT)")]),
    ("base", "Base", [("1.4", "SD1.4"), ("1.5", "SD1.5"), ("2.1", "SD2.1")]),
    ("mod", "Edits", [("none", "None"), ("guidance", "Guidance"), ("clip", "CLIP-enc"),
                      ("text", "Text-enc"), ("unet", "U-Net")]),
]


def read_prompts(fname):
    p = PROMPT_DIR / fname
    if not p.exists():
        return []
    out = []
    for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
        t = ln.strip()
        if t and not t.startswith("#"):
            out.append(t)
    return out


def rel(path):
    return os.path.relpath(path, OUT.parent).replace(os.sep, "/")


def _load_json(p):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_violence():
    out = {}
    j = _load_json(VIOLENCE_JSON) if VIOLENCE_JSON.exists() else None
    if j:
        for lbl, d in (j.get("models") or {}).items():
            v = d.get("asr_violence_mean")
            if v is not None:
                out[lbl] = v
    return out


def load_violence50():
    """key -> asr_violence_mean from the 50-prompt violence subset (violence_q16_smoke.json)."""
    out = {}
    j = _load_json(VIOLENCE50_JSON) if VIOLENCE50_JSON.exists() else None
    if j:
        for lbl, d in (j.get("models") or {}).items():
            v = d.get("asr_violence_mean")
            if v is not None:
                out[lbl] = v
    return out


def load_cost():
    """key -> {gpu_hours, trainable_params_M, training_free, gpu} from env-aware train_cost.json."""
    out = {}
    j = _load_json(TRAINCOST_JSON) if TRAINCOST_JSON.exists() else None
    if j:
        for lbl, d in (j.get("models") or {}).items():
            out[lbl] = d
    return out


def load_style():
    """key -> style_img2raw (CLIP cos(model_img, raw_img); higher = Van Gogh style preserved)."""
    out = {}
    j = _load_json(STYLE_JSON) if STYLE_JSON.exists() else None
    if j:
        for lbl, d in (j.get("models") or {}).items():
            v = d.get("style_img2raw")
            if v is not None:
                out[lbl] = v
    return out


def load_fullset():
    """key -> {a8:{label:val}, m8, m4}. Frozen full-set nudity ASR (score>0.3).

    Per-attack + mean use the 8-label rule (a8 / m8); m4 is the 4-label (FCF
    exposed-only) mean at the same 0.3 threshold. raw/fcf come from
    fullset_eval.json (asr_ours8 / asr_fcf4); every other Table-A method from
    fullset_all.json (ours8_p03 / fcf4_p03).
    """
    out = {}
    je = _load_json(FULLSET_EVAL) if FULLSET_EVAL.exists() else None
    for k, d in ((je or {}).get("models") or {}).items():
        a8 = {a: v.get("asr_ours8") for a, v in (d.get("attacks") or {}).items()}
        out[k] = {"a8": a8, "m8": d.get("asr_ours8_mean"), "m4": d.get("asr_fcf4_mean")}
    ja = _load_json(FULLSET_ALL) if FULLSET_ALL.exists() else None
    for k, d in ((ja or {}).get("models") or {}).items():
        if "error" in d:
            continue
        a8 = {a: v.get("ours8_p03") for a, v in (d.get("attacks") or {}).items()}
        out[k] = {"a8": a8, "m8": d.get("ours8_p03_mean"), "m4": d.get("fcf4_p03_mean")}
    return out


def load_metrics():
    """key -> {fs_asr, fs_mean8, fs_mean4, leg_asr, leg_mean, fid, coco_clip, violence}.

    fs_* = frozen FULL-SET nudity ASR; leg_* = legacy 50-prompt re-score
    (eval/outputs/<key>/metrics.json). COCO-FID/CLIP from coco_metrics.json,
    violence from violence_q16.json. People-ASR intentionally dropped.
    """
    viol = load_violence()
    viol50 = load_violence50()
    sty = load_style()
    cost = load_cost()
    fs = load_fullset()
    data = {}
    for _, key, *_ in MODELS:
        d = {"fs_asr": {}, "fs_mean8": None, "fs_mean4": None,
             "leg_asr": {}, "leg_mean": None,
             "fid": None, "coco_clip": None, "violence": viol.get(key),
             "violence50": viol50.get(key), "style": sty.get(key),
             "cost": cost.get(key)}
        m = _load_json(FS_ROOT / key / "metrics.json")
        if m:
            d["leg_asr"] = m.get("asr", {}) or {}
            d["leg_mean"] = m.get("asr_mean")
        f = fs.get(key)
        if f:
            d["fs_asr"] = {a: v for a, v in f["a8"].items() if v is not None}
            d["fs_mean8"] = f["m8"]
            d["fs_mean4"] = f["m4"]
        c = _load_json(FS_ROOT / key / "coco_metrics.json")
        if c:
            d["fid"] = c.get("coco_fid")
            d["coco_clip"] = c.get("coco_clip")
        data[key] = d
    return data


def heat(v):
    """0 (green/good) -> 100 (red/bad) background for an ASR-like percentage."""
    t = max(0.0, min(1.0, float(v) / 100.0))
    return "background:hsl({0:.0f},55%,26%)".format(120 * (1 - t))


def asr_cell(v, bold=False):
    cls = "num b" if bold else "num"
    if v is None:
        return '<td class="{0} pend">—</td>'.format(cls)
    return '<td class="{0}" data-v="{2:.2f}" style="{1}">{2:.1f}</td>'.format(cls, heat(v), float(v))


def num_cell(v):
    if v is None:
        return '<td class="num muted">—</td>'
    return '<td class="num muted">{0:.1f}</td>'.format(float(v))


def ret_cell(v):
    """Retention/locality cell: CLIP cos in [~0.6,1.0] shown x100; HIGHER is better (green)."""
    if v is None:
        return '<td class="num pend">—</td>'
    pct = 100.0 * float(v)
    t = max(0.0, min(1.0, (pct - 70.0) / 30.0))          # 70->red .. 100->green
    return '<td class="num" data-v="{1:.2f}" style="background:hsl({0:.0f},55%,26%)">{1:.1f}</td>'.format(
        120 * t, pct)


def cost_cell(c):
    """Training cost cell: green = cheaper. 'free' = no local training (raw/SLD/safe-neg)."""
    if not c:
        return '<td class="num pend">—</td>'
    if c.get("training_free"):
        return '<td class="num" title="no local training" style="background:hsl(120,45%,24%)">free</td>'
    gh = c.get("gpu_hours")
    if gh is None:
        return '<td class="num muted">—</td>'
    pm = c.get("trainable_params_M")
    lbl = "{0:.2f}h".format(float(gh)) + ("/{0:.0f}M".format(float(pm)) if pm else "")
    t = max(0.0, min(1.0, float(gh) / 3.0))
    bg = "background:hsl({0:.0f},55%,26%)".format(120 * (1 - t))
    return '<td class="num" data-v="{0:.3f}" title="{1}" style="{2}">{3}</td>'.format(
        float(gh), html.escape(str(c.get("gpu", ""))), bg, lbl)


def _mh(label, key, group, base, mod):
    return '<th class="mh" title="SD{0} / {1}">{2}<span class="tag t-{3}">{3}</span></th>'.format(
        base, mod, html.escape(label), group)


def _tr(key, group, base, mod, tds):
    pin = ' data-pin="1"' if group == "ref" else ""
    return '<tr data-key="{0}" data-group="{1}" data-base="{2}" data-mod="{3}"{4}>{5}</tr>'.format(
        key, group, base, mod, pin, "".join(tds))


def _table(head, body):
    return ('<table class="qt"><thead><tr>' + "".join(head) +
            '</tr></thead><tbody>' + "".join(body) + '</tbody></table>')


def fullset_table(M):
    head = ['<th class="mh">Model</th>']
    head += ['<th>{0}&nbsp;<span class="ar">&darr;</span></th>'.format(html.escape(a)) for a, _, _ in ATTACKS]
    head += ['<th>ASR&nbsp;mean&nbsp;<span class="ar">&darr;</span><br><span class="sub">8-lab</span></th>',
             '<th>ASR&nbsp;mean&nbsp;<span class="ar">&darr;</span><br><span class="sub">4-lab</span></th>',
             '<th class="muted">COCO-FID&nbsp;<span class="ar">&darr;</span></th>',
             '<th class="muted">COCO-CLIP&nbsp;<span class="ar">&uarr;</span></th>',
             '<th>Violence&nbsp;Q16&nbsp;<span class="ar">&darr;</span></th>',
             '<th>VanGogh&nbsp;<span class="ar">&uarr;</span><br><span class="sub">retain</span></th>',
             '<th>Train&nbsp;<span class="ar">&darr;</span><br><span class="sub">GPU-h</span></th>']
    body = []
    for label, key, group, base, mod in MODELS:
        d = M[key]
        tds = [_mh(label, key, group, base, mod)]
        tds += [asr_cell(d["fs_asr"].get(a)) for a, _, _ in ATTACKS]
        tds.append(asr_cell(d["fs_mean8"], bold=True))
        tds.append(asr_cell(d["fs_mean4"], bold=True))
        tds.append(num_cell(d["fid"]))
        tds.append(num_cell(d["coco_clip"]))
        tds.append(asr_cell(d["violence"]))
        tds.append(ret_cell(d["style"]))
        tds.append(cost_cell(d["cost"]))
        body.append(_tr(key, group, base, mod, tds))
    return _table(head, body)


def legacy_table(M):
    head = ['<th class="mh">Model</th>']
    head += ['<th>{0}&nbsp;<span class="ar">&darr;</span></th>'.format(html.escape(a)) for a, _, _ in ATTACKS]
    head += ['<th>ASR&nbsp;mean&nbsp;<span class="ar">&darr;</span></th>',
             '<th>Violence-50&nbsp;<span class="ar">&darr;</span></th>',
             '<th>VanGogh&nbsp;<span class="ar">&uarr;</span><br><span class="sub">retain</span></th>']
    body = []
    for label, key, group, base, mod in MODELS:
        d = M[key]
        tds = [_mh(label, key, group, base, mod)]
        tds += [asr_cell(d["leg_asr"].get(a)) for a, _, _ in ATTACKS]
        tds.append(asr_cell(d["leg_mean"], bold=True))
        tds.append(asr_cell(d["violence50"]))
        tds.append(ret_cell(d["style"]))
        body.append(_tr(key, group, base, mod, tds))
    return _table(head, body)


def chip_bar():
    out = []
    for dim, lbl, vals in FILTERS:
        out.append('<span class="fgl">{0}:</span>'.format(html.escape(lbl)))
        for v, t in vals:
            out.append('<button class="chip" data-dim="{0}" data-val="{1}">{2}</button>'.format(
                dim, v, html.escape(t)))
        out.append('<span class="sep"></span>')
    return "".join(out)


CSS = """
:root{--bg:#111315;--panel:#191c20;--panel2:#22262b;--text:#eff2f4;--muted:#a8b0b8;--line:#343a42;--accent:#5ec6a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:30;background:rgba(17,19,21,.97);border-bottom:1px solid var(--line);padding:10px 16px}
h1{margin:0 0 8px;font-size:19px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.toolbar.filt{margin-top:8px}
button,input,label.t{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer}button.active{border-color:var(--accent);background:#17352f}
label.t{display:inline-flex;gap:6px;align-items:center}
.live{color:var(--accent);font-size:12px;margin-left:6px}
.fgl{color:var(--muted);font-size:12px;margin-left:6px}
.chip{padding:5px 10px;font-size:12px;border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:14px;cursor:pointer}
.chip.on{border-color:var(--accent);background:#17352f;color:#cfeee4}
.sep{display:inline-block;width:1px;height:18px;background:var(--line);margin:0 2px;vertical-align:middle}
.vc{color:var(--muted);font-size:12px;margin-left:6px}
.sec{padding:14px 16px 30px}.sec h2{font-size:17px;margin:4px 0 10px}.sec h2 span{color:var(--muted);font-weight:400;font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
/* quantitative table */
.qt{border-collapse:collapse;width:100%;font-size:13px}
.qt th,.qt td{border:1px solid var(--line);padding:6px 9px;text-align:center;white-space:nowrap}
.qt thead th{position:sticky;top:0;background:var(--panel2);z-index:6}
.qt th.mh{position:sticky;left:0;background:var(--panel2);text-align:left;z-index:7;min-width:178px}
.qt thead th.mh{z-index:8}
.qt td.num{font-variant-numeric:tabular-nums}
.qt td.num.b{font-weight:800}
.qt th .sub{font-weight:400;color:var(--muted);font-size:10px}
.qt th .ar{color:var(--accent);font-size:11px;font-weight:700}
.qt td.muted{color:var(--muted)}
.qt td.pend{color:#5a626b}
.qt tbody tr:hover td,.qt tbody tr:hover th.mh{outline:1px solid var(--accent);outline-offset:-1px}
.tag{display:inline-block;margin-left:8px;font-size:10px;padding:1px 6px;border-radius:9px;vertical-align:middle}
.tag.t-ref{background:#26323e;color:#86b6e0}.tag.t-baseline{background:#2c2f34;color:#a8b0b8}
.tag.t-core{background:#243a36;color:#73c7b0}.tag.t-novel{background:#1f3a2a;color:#6ad08e}
.legend{color:var(--muted);font-size:12px;margin:8px 2px 0}
.legend b{color:var(--text)}
/* image gallery */
.grid{display:grid;grid-template-columns:240px repeat(var(--mc),minmax(150px,1fr))}
.c{border-right:1px solid var(--line);border-bottom:1px solid var(--line);background:var(--panel);padding:6px}
.c.h{background:var(--panel2);font-weight:700;position:sticky;top:0;min-height:40px}
.c.pc{position:sticky;left:0;z-index:5;background:var(--panel2);color:var(--muted);font-size:12px}
.c.pc b{color:var(--text)}
img{width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:5px;background:#0b0c0d;display:block;transition:filter .12s}
body.blur img{filter:blur(16px) saturate(.5)}
img.pending{outline:1px dashed var(--line)}
.lbl{color:var(--muted);font-size:10px;margin-top:3px;text-align:center}
.hidden{display:none!important}
"""

JS_TMPL = """
__META__
let auto=true, pin=true;
const FILT={group:new Set(),base:new Set(),mod:new Set()};
function modelVisible(m){
  if(pin && m.group==='ref') return true;
  let dims=['group','base','mod'];
  for(let i=0;i<dims.length;i++){var s=FILT[dims[i]];if(s.size && !s.has(m[dims[i]]))return false;}
  return true;
}
function applyFilter(){
  let vis=0;
  ORDER.forEach(function(k){
    var show=modelVisible(META[k]); if(show)vis++;
    document.querySelectorAll('tr[data-key="'+k+'"]').forEach(function(r){r.classList.toggle('hidden',!show);});
    document.querySelectorAll('.m-'+k).forEach(function(c){c.classList.toggle('hidden',!show);});
  });
  document.querySelectorAll('.grid').forEach(function(g){g.style.setProperty('--mc',vis);});
  var vc=document.getElementById('viscount'); if(vc)vc.textContent=vis;
}
function retry(){
  if(!auto)return;
  document.querySelectorAll('img[data-src]').forEach(function(im){
    if(im.naturalWidth===0){im.classList.add('pending');im.src=im.getAttribute('data-src')+'?t='+Date.now();}
    else{im.classList.remove('pending');}
  });
}
setInterval(retry,25000);
document.addEventListener('DOMContentLoaded',function(){
  var btns=[].slice.call(document.querySelectorAll('[data-ab]'));
  var secs=[].slice.call(document.querySelectorAll('[data-as]'));
  btns.forEach(function(b){b.addEventListener('click',function(){
    btns.forEach(function(x){x.classList.toggle('active',x===b);});
    var k=b.dataset.ab;
    secs.forEach(function(s){s.classList.toggle('hidden',k!=='all'&&s.dataset.as!==k);});
  });});
  document.querySelectorAll('.chip').forEach(function(c){c.addEventListener('click',function(){
    var dim=c.dataset.dim, val=c.dataset.val;
    if(FILT[dim].has(val)){FILT[dim].delete(val);c.classList.remove('on');}
    else{FILT[dim].add(val);c.classList.add('on');}
    applyFilter();
  });});
  document.getElementById('pin').addEventListener('change',function(e){pin=e.target.checked;applyFilter();});
  document.getElementById('reset').addEventListener('click',function(){
    ['group','base','mod'].forEach(function(d){FILT[d].clear();});
    document.querySelectorAll('.chip.on').forEach(function(c){c.classList.remove('on');});
    applyFilter();
  });
  document.getElementById('blur').addEventListener('change',function(e){document.body.classList.toggle('blur',e.target.checked);});
  document.getElementById('autob').addEventListener('click',function(e){auto=!auto;e.target.textContent=auto?'⏸ Auto-refresh: ON':'▶ Auto-refresh: OFF';if(auto)retry();});
  document.getElementById('now').addEventListener('click',retry);
  applyFilter();
});
"""


def build(rows_cap):
    M = load_metrics()
    n_models = len(MODELS)
    parts = ['<!doctype html><html lang="ko"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>SD Unlearning - Live Gallery</title><style>', CSS, '</style></head><body class="blur">']

    abtn = ['<button class="active" data-ab="all">All</button>']
    abtn += ['<button data-ab="{0}">{0}</button>'.format(html.escape(a)) for a, _, _ in ATTACKS]
    parts.append(
        '<header><h1>SD Unlearning - Live Gallery '
        '<span class="live">auto-fills as images are generated</span></h1>'
        '<div class="toolbar">' + "".join(abtn) +
        '<label class="t"><input id="blur" type="checkbox" checked> blur</label>'
        '<button id="autob">⏸ Auto-refresh: ON</button>'
        '<button id="now">↻ Refresh now</button></div>'
        '<div class="toolbar filt">' + chip_bar() +
        '<label class="t"><input id="pin" type="checkbox" checked> Pin Raw&nbsp;SD</label>'
        '<button id="reset">Reset filters</button>'
        '<span class="vc">visible <b id="viscount">{0}</b>/{0} models</span>'.format(n_models) +
        '</div></header><main>')

    # 1) frozen full-set table (8-lab + 4-lab means)
    parts.append(
        '<section class="sec"><h2>Quantitative results '
        '<span>(nudity ASR % &middot; frozen full-set &middot; both means score&gt;0.3, '
        '8-lab=our rule / 4-lab=FCF exposed-only &middot; lower is better &middot; '
        'heatmap green=low/red=high &middot; &ndash; = pending)</span></h2><div class="wrap">')
    parts.append(fullset_table(M))
    parts.append('</div><div class="legend">'
                 '<b>ASR mean 8-lab</b>=our strict rule (4 exposed + covered + buttocks) &middot; '
                 '<b>ASR mean 4-lab</b>=FCF rule (4 exposed labels only) &middot; per-attack cells are 8-lab &middot; '
                 '<b>frozen full-set</b>: I2P&nbsp;931 / RaB&nbsp;95 / RaB(Re)&nbsp;95 / P4D&nbsp;361 / UDA&nbsp;142 &middot; '
                 '<b>COCO-FID</b>/<b>CLIP</b>=utility &middot; <b>Violence Q16</b>=off-target violence ASR &middot; '
                 '<b>VanGogh retain</b>=CLIP image&harr;raw-SD similarity on 50 Van Gogh prompts (&uarr; higher=style preserved, locality). '
                 'FCF looks worse on 8-lab than 4-lab because it only targets the 4 exposed labels.'
                 '</div></section>')

    # 1b) legacy 50-prompt table (reference)
    parts.append(
        '<section class="sec"><h2>Legacy results '
        '<span>(50-prompt harness, 8-label score&gt;0.3 &middot; the earlier per-attack re-score, '
        'kept for reference)</span></h2><div class="wrap">')
    parts.append(legacy_table(M))
    parts.append('</div><div class="legend">'
                 'Same NudeNet 8-label score&gt;0.3 rule as the table above, but only '
                 '<b>50 prompts/attack</b> (the pre-full-set re-score). '
                 'Models without a 50-prompt metrics.json show &ndash;.'
                 '</div></section>')

    # 2) live image gallery, one section per attack
    for albl, sub, pf in ATTACKS:
        prompts = read_prompts(pf)
        n = len(prompts)
        rows = n if rows_cap == 0 else min(rows_cap, n if n else rows_cap)
        parts.append('<section class="sec" data-as="{0}"><h2>{0} '
                     '<span>({1}, {2} prompts, showing {3})</span></h2><div class="wrap">'
                     '<div class="grid" style="--mc:{4}">'.format(
                         html.escape(albl), html.escape(sub), n, rows, n_models))
        parts.append('<div class="c h pc">Prompt</div>')
        for mlbl, key, *_ in MODELS:
            parts.append('<div class="c h m-{0}">{1}</div>'.format(key, html.escape(mlbl)))
        for i in range(rows):
            ptxt = prompts[i] if i < len(prompts) else ""
            ptxt = (ptxt[:140] + "…") if len(ptxt) > 140 else ptxt
            parts.append('<div class="c pc"><b>#{0:04d}</b><br>{1}</div>'.format(i, html.escape(ptxt)))
            for mlbl, key, *_ in MODELS:
                src = rel(FS_ROOT / "{0}_fs".format(key) / sub / "{0:04d}_00.png".format(i))
                parts.append(
                    '<div class="c m-{3}"><a href="{0}" target="_blank" rel="noopener">'
                    '<img loading="lazy" class="pending" data-src="{0}" src="{0}" alt="{1} {2:04d}">'
                    '</a><div class="lbl">{1}</div></div>'.format(src, html.escape(mlbl), i, key))
        parts.append('</div></div></section>')

    meta = "const META=" + json.dumps(
        {key: {"label": label, "group": g, "base": b, "mod": m}
         for label, key, g, b, m in MODELS}) + ";const ORDER=" + json.dumps(
        [key for _, key, *_ in MODELS]) + ";"
    parts.append('</main><script>' + JS_TMPL.replace("__META__", meta) + '</script></body></html>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=80, help="rows/attack (0=all)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.write_text(build(a.rows), encoding="utf-8")
    n_metrics = sum(1 for _, k, *_ in MODELS if (FS_ROOT / k / "metrics.json").exists())
    print("wrote {0} ({1} KB) | {2} models ({3} with metrics) x {4} rows x {5} attacks".format(
        out, out.stat().st_size // 1024, len(MODELS), n_metrics, a.rows or "all", len(ATTACKS)))


if __name__ == "__main__":
    main()
