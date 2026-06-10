"""Live full-set qualitative gallery over eval/outputs/<model>_fs/<attack>/.

Unlike build_gallery.py (old 50-prompt eval layout + ASR table), this targets the
FULL-SET images that eval_fullset_all.py writes to eval/outputs/<model>_fs/. Every
cell emits an <img> pointing at the FINAL path even if the file does not exist yet;
a JS timer retries not-yet-loaded images (cache-busted) every 25s, so the gallery
fills in LIVE as the background job generates each image. Open
compare/comparison_gallery_live.html and leave it open while P3-ext runs.

Run:  python compare/build_live_gallery.py [--rows 80] [--out PATH]
      --rows 0  -> all prompts per attack (large page)
"""
from __future__ import annotations
import argparse
import html
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "compare" / "comparison_gallery_live.html"
FS_ROOT = REPO / "eval" / "outputs"
PROMPT_DIR = REPO / "models" / "fcf" / "data" / "eval"

# (display label, _fs key)
MODELS = [
    ("Raw SD v1.4", "raw_v14"), ("FCF-P (official)", "fcf_p_official"),
    ("FCF-E (official)", "fcf_e_official"), ("ODACE v3", "odace_v3"),
    ("ODACE v1.5", "odace_v15"), ("ODACE earlier", "odace_v2"),
    ("Spherical+OT", "sph_ot"), ("LSSE+PLU", "lsse_plu"), ("LSSE+PLU+W2", "lsse_plu_w2"),
    ("Vanilla LSSE", "vanilla_lsse"), ("DACE", "dace_v2"), ("DACE+PLU", "dace_plu"),
    ("ESD-u", "esd_u"), ("Safe-CLIP", "safeclip"), ("Safe-neg", "safe_neg"),
    ("SLD-Medium", "sld_medium"), ("SLD-Strong", "sld_strong"), ("SLD-Max", "sld_max"),
    ("SD2.1-base", "sd21base"), ("Raw SD v1.5", "raw_v15"),
]
# (label, _fs subdir, prompt file)
ATTACKS = [
    ("I2P", "i2p", "i2p_nudity.txt"),
    ("Ring-A-Bell", "ring_a_bell", "ring_a_bell_nudity.txt"),
    ("Ring-A-Bell(Re)", "ring_a_bell_re", "ring_a_bell_re_nudity.txt"),
    ("P4D", "p4d", "p4d_nudity.txt"),
    ("UnlearnDiffAtk", "unlearndiffatk", "unlearnDiffAtk_nudity.txt"),
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


CSS = """
:root{--bg:#111315;--panel:#191c20;--panel2:#22262b;--text:#eff2f4;--muted:#a8b0b8;--line:#343a42;--accent:#5ec6a8}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 ui-sans-serif,system-ui,"Segoe UI",sans-serif}
header{position:sticky;top:0;z-index:20;background:rgba(17,19,21,.97);border-bottom:1px solid var(--line);padding:12px 16px}
h1{margin:0 0 8px;font-size:19px}
.toolbar{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
button,input,label.t{border:1px solid var(--line);background:var(--panel);color:var(--text);border-radius:6px;padding:7px 10px;font:inherit}
button{cursor:pointer}button.active{border-color:var(--accent);background:#17352f}
label.t{display:inline-flex;gap:6px;align-items:center}
.live{color:var(--accent);font-size:12px;margin-left:6px}
.sec{padding:14px 16px 30px}.sec h2{font-size:17px;margin:4px 0 10px}.sec h2 span{color:var(--muted);font-weight:400;font-size:13px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
.grid{display:grid;grid-template-columns:240px repeat(var(--mc),minmax(150px,1fr));min-width:calc(240px + var(--mc)*150px)}
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

JS = """
let auto=true;
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
  document.getElementById('blur').addEventListener('change',function(e){document.body.classList.toggle('blur',e.target.checked);});
  document.getElementById('autob').addEventListener('click',function(e){auto=!auto;e.target.textContent=auto?'⏸ Auto-refresh: ON':'▶ Auto-refresh: OFF';if(auto)retry();});
  document.getElementById('now').addEventListener('click',retry);
});
"""


def build(rows_cap):
    parts = ['<!doctype html><html lang="ko"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>SD Unlearning - Live Gallery</title><style>', CSS, '</style></head><body class="blur">']
    abtn = ['<button class="active" data-ab="all">All</button>']
    abtn += ['<button data-ab="{0}">{0}</button>'.format(html.escape(k)) for k, _, _ in ATTACKS]
    parts.append('<header><h1>SD Unlearning - Live Gallery '
                 '<span class="live">auto-fills as images are generated</span></h1>'
                 '<div class="toolbar">' + "".join(abtn) +
                 '<label class="t"><input id="blur" type="checkbox" checked> blur</label>'
                 '<button id="autob">⏸ Auto-refresh: ON</button>'
                 '<button id="now">↻ Refresh now</button></div></header><main>')
    for albl, sub, pf in ATTACKS:
        prompts = read_prompts(pf)
        n = len(prompts)
        rows = n if rows_cap == 0 else min(rows_cap, n if n else rows_cap)
        parts.append('<section class="sec" data-as="{0}"><h2>{0} '
                     '<span>({1}, {2} prompts, showing {3})</span></h2><div class="wrap">'
                     '<div class="grid" style="--mc:{4}">'.format(
                         html.escape(albl), html.escape(sub), n, rows, len(MODELS)))
        parts.append('<div class="c h pc">Prompt</div>')
        for mlbl, _ in MODELS:
            parts.append('<div class="c h">{0}</div>'.format(html.escape(mlbl)))
        for i in range(rows):
            ptxt = prompts[i] if i < len(prompts) else ""
            ptxt = (ptxt[:140] + "…") if len(ptxt) > 140 else ptxt
            parts.append('<div class="c pc"><b>#{0:04d}</b><br>{1}</div>'.format(i, html.escape(ptxt)))
            for mlbl, key in MODELS:
                src = rel(FS_ROOT / "{0}_fs".format(key) / sub / "{0:04d}_00.png".format(i))
                parts.append(
                    '<div class="c"><a href="{0}" target="_blank" rel="noopener">'
                    '<img loading="lazy" class="pending" data-src="{0}" src="{0}" alt="{1} {2:04d}">'
                    '</a><div class="lbl">{1}</div></div>'.format(src, html.escape(mlbl), i))
        parts.append('</div></div></section>')
    parts.append('</main><script>' + JS + '</script></body></html>')
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=80, help="rows/attack (0=all)")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    out = Path(a.out)
    out.write_text(build(a.rows), encoding="utf-8")
    print("wrote {0} ({1} KB), {2} models x {3} rows x {4} attacks".format(
        out, out.stat().st_size // 1024, len(MODELS), a.rows or "all", len(ATTACKS)))


if __name__ == "__main__":
    main()
