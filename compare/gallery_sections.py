"""Extra narrative sections for the live gallery (imported by build_live_gallery.py).

Kept in a separate module so the main builder stays focused on the data tables and
the primary image gallery. All renderers are pure functions that take a `Ctx`
(dependency-injected primitives from the main builder) plus already-loaded data,
so there is no circular import and nothing here touches the filesystem beyond the
image strip listing.

Sections:
  E. headline_section       - hero "key findings" cards (top of page)
  A. ood_image_strip        - Ring-A-Bell vs I2P image proof of OOD collapse
  D. taxonomy_section       - push-away vs redirect x TE vs UNet 2x2 map
  C. rpgrt_curve_section    - RPG-RT DPO adaptation curve (asr_query per iter)
  F. transfer_heatmap_section - cross-concept transfer (nudity <-> violence)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Ctx:
    """Primitives injected from build_live_gallery so this module needs no back-import."""
    rel: Callable          # Path -> relative url string
    fs_root: Path          # eval/outputs
    html: object           # the stdlib `html` module (for escape)
    table: Callable        # _table(head, body)
    mh: Callable           # _mh(label, key, group, base, mod) -> <th>
    asr_cell: Callable     # asr_cell(v, bold=False) -> heat-colored <td>
    ret_cell: Callable     # ret_cell(v) -> 0..1 retention <td>
    num_cell: Callable     # num_cell(v) -> muted numeric <td>
    model_info: Callable   # key -> (label, group, base, mod)
    labels: dict           # key -> display label


def _fmt(v, d: int = 1, suf: str = "") -> str:
    if v is None:
        return "&#8212;"
    return "{0:.{1}f}{2}".format(float(v), d, suf)


def _pct(v) -> str:
    return "&#8212;" if v is None else "{0:.0f}%".format(float(v) * 100.0)


# ---------------------------------------------------------------- E. headline card
def headline_section(ctx: Ctx, M: dict) -> str:
    """Hero band: the four take-away claims with live flagship numbers pulled from M."""
    g = lambda k, f: (M.get(k, {}) or {}).get(f)
    ring_collapse = g("odace_v3", "coh_ring")
    ring_fix = g("odace_benign_n1", "coh_ring")
    te_asr = g("lsse_r2q_ab", "fs_mean8")
    te_clip = g("lsse_r2q_ab", "coco_clip")
    unet_asr = g("odace_v3", "fs_mean8")

    cards = [
        ("accent",
         "{0} &rarr; {1}".format(_pct(ring_collapse), _pct(ring_fix)),
         "OOD person-coherence (discovery)",
         "On the Ring-A-Bell OOD attack, <b>push-away</b> erasure collapses to non-human "
         "garbage (person_prob&rarr;0), so its low ASR is <b>collapse, not erasure</b>. "
         "<b>Redirect-to-benign</b> restores coherence &mdash; a fix that is mechanism, not a "
         "hyper-parameter (confirmed by the no-OOD-aug ablation)."),
        ("",
         "{0}%".format(_fmt(te_asr)),
         "LSSE CAP-CNP &mdash; text-encoder",
         "8-lab ASR at COCO-CLIP {0}. Dominates SLERP-OT (15.6) and FCF-P (3.7) on "
         "<b>both</b> the safety and utility axes &mdash; strongest text-encoder-only point.".format(_fmt(te_clip))),
        ("",
         "{0}%".format(_fmt(unet_asr)),
         "ODACE &mdash; UNet cross-attn",
         "Output-grounded editing; beats every baseline (ESD-u&nbsp;21.6 / Safe-CLIP&nbsp;44 / "
         "SLD&nbsp;45&ndash;62). Its benign-anchor variant is OOD-coherent (ring&nbsp;1.00)."),
        ("",
         "gap &asymp; 0",
         "Adaptive red-team (RPG-RT)",
         "Against a DPO-fine-tuned Vicuna-7B attacker, redirect defenses (SLERP-OT, ODACE "
         "benign) are <b>unmovable</b> (iter0&rarr;best gap&nbsp;0), while raw explodes 52.5&rarr;75."),
    ]
    cells = []
    for cls, num, lbl, desc in cards:
        cells.append(
            '<div class="hcard {0}"><div class="hnum">{1}</div>'
            '<div class="hlbl">{2}</div><div class="hdesc">{3}</div></div>'.format(cls, num, lbl, desc))
    return ('<section class="sec"><h2>Key findings '
            '<span>(single-concept nudity erasure on SD1.4 &middot; lower ASR is only trustworthy '
            'when OOD coherence stays high)</span></h2>'
            '<div class="hero">' + "".join(cells) + '</div></section>')


# ------------------------------------------------------------- A. OOD image strip
_STRIP_KEYS = [
    ("raw_v14", "reference (un-erased)"),
    ("lsse_r2q_ab", "push-away &middot; text-enc"),
    ("odace_v3", "push-away &middot; UNet"),
    ("sph_ot", "redirect &middot; text-enc"),
    ("odace_benign_n1", "redirect &middot; UNet"),
]


def _first_pngs(d: Path, k: int) -> list:
    if not d.exists():
        return []
    return sorted(d.glob("*.png"))[:k]


def ood_image_strip(ctx: Ctx, M: dict, k_ring: int = 6, k_i2p: int = 3) -> str:
    esc = ctx.html.escape
    rows = []
    for key, tag in _STRIP_KEYS:
        base = ctx.fs_root / "{0}_fs".format(key)
        ring = _first_pngs(base / "ring_a_bell", k_ring)
        i2p = _first_pngs(base / "i2p", k_i2p)
        if not ring and not i2p:
            continue
        d = M.get(key, {}) or {}
        r = d.get("coh_ring")
        if r is None:
            badge, vcls = "", ""
        elif r < 0.4:
            badge = '<span class="badge col">collapse &middot; ring {0}</span>'.format(_pct(r))
            vcls = "v-col"
        elif r >= 0.7:
            badge = '<span class="badge coh">coherent &middot; ring {0}</span>'.format(_pct(r))
            vcls = "v-coh"
        else:
            badge = '<span class="badge">partial &middot; ring {0}</span>'.format(_pct(r))
            vcls = ""

        def imgs(paths):
            out = []
            for p in paths:
                u = ctx.rel(p)
                out.append('<a href="{0}" target="_blank" rel="noopener">'
                           '<img loading="lazy" src="{0}" alt=""></a>'.format(u))
            return '<div class="oodimgs">' + "".join(out) + '</div>'

        lbl = esc(ctx.labels.get(key, key))
        rows.append(
            '<div class="oodrow"><div class="oodlab"><b class="{0}">{1}</b><br>'
            '<span class="muted" style="font-size:11px">{2}</span><br>{3}</div>{4}{5}</div>'.format(
                vcls, lbl, tag, badge, imgs(ring), imgs(i2p)))
    if not rows:
        return ""
    hd = ('<div class="oodhd"><div>Model</div>'
          '<div>Ring-A-Bell (OOD attack)</div><div>I2P (natural control)</div></div>')
    note = ('<div class="legend"><b>Visual proof of collapse.</b> Read left&rarr;right: an un-erased '
            'reference, two <b>push-away</b> models that render incoherent non-human output on the OOD '
            'Ring-A-Bell attack (their ASR is near-zero for the <i>wrong</i> reason), then two '
            '<b>redirect-to-benign</b> models that stay on-manifold and produce coherent <i>safe</i> '
            'images. The I2P control shows all models render normally on in-distribution prompts. '
            'Images inherit the page-level blur toggle.</div>')
    return ('<section class="sec"><h2>OOD collapse &mdash; visual proof '
            '<span>(Ring-A-Bell samples vs I2P control &middot; toggle blur off to inspect)</span></h2>'
            '<div class="wrap" style="padding:0 12px 10px">' + hd + "".join(rows) + '</div>'
            + note + '</section>')


# ----------------------------------------------------------------- D. taxonomy 2x2
# (space, mechanism) -> [(key, note)]; mechanism col order = push-away, redirect
_TAXO = {
    ("Text-encoder", "push"): [("lsse_r2q_ab", "flagship"), ("lsse_r2q_a", "max-forget"),
                               ("lsse_capcnp_zero", "R2"), ("lsse_geo_e2", "geodesic")],
    ("Text-encoder", "redirect"): [("sph_ot", "SLERP-OT"), ("fcf_p_official", "FCF-P"),
                                   ("fcf_e_official", "FCF-E")],
    ("UNet x-attn", "push"): [("odace_v3", "neg-guide"), ("esd_u", "ESD-u")],
    ("UNet x-attn", "redirect"): [("odace_benign_n1", "benign-neg"), ("odace_benign", "benign-anchor")],
}


def _mchip(ctx: Ctx, key: str, note: str, M: dict) -> str:
    r = (M.get(key, {}) or {}).get("coh_ring")
    cls = "mchip"
    if r is not None:
        cls += " col" if r < 0.4 else (" coh" if r >= 0.7 else " par")
    lbl = ctx.html.escape(ctx.labels.get(key, key)).replace("LSSE+CAP-CNP ", "").replace("LSSE ", "")
    ttl = "ring {0}".format(_pct(r)) if r is not None else "no OOD probe"
    return '<span class="{0}" title="{1}">{2}<span class="muted"> &middot; {3}</span></span>'.format(
        cls, ttl, lbl, ctx.html.escape(note))


def taxonomy_section(ctx: Ctx, M: dict) -> str:
    def cell(space, mech):
        chips = "".join(_mchip(ctx, k, n, M) for k, n in _TAXO[(space, mech)])
        return '<div class="tcell">{0}</div>'.format(chips)
    grid = [
        '<div class="thd"></div>',
        '<div class="thd">Push-away erasure<br><span class="muted" style="font-weight:400">'
        'move <i>off</i> the concept (geodesic / negative-guidance)</span></div>',
        '<div class="thd">Redirect-to-benign<br><span class="muted" style="font-weight:400">'
        'move <i>to</i> a safe anchor (SLERP-OT / benign target)</span></div>',
        '<div class="rhd">Text-encoder</div>',
        cell("Text-encoder", "push"), cell("Text-encoder", "redirect"),
        '<div class="rhd">UNet x-attn</div>',
        cell("UNet x-attn", "push"), cell("UNet x-attn", "redirect"),
    ]
    note = ('<div class="legend"><b>Why the two columns behave differently.</b> Push-away objectives '
            'leave the generative manifold on OOD inputs &rarr; collapse '
            '(<span style="color:#e08a8a">red</span> chips = low OOD coherence). Redirect-to-benign '
            'objectives keep outputs on-manifold &rarr; coherent erasure '
            '(<span style="color:#7cd0a4">green</span>). Chip color = Ring-A-Bell person_prob '
            '(&lt;0.4 collapse, &ge;0.7 coherent). This holds across <b>both</b> intervention spaces, '
            'so the fix is the objective geometry, not the edit site.</div>')
    return ('<section class="sec"><h2>Method taxonomy '
            '<span>(intervention space &times; erasure mechanism &middot; color = OOD coherence)</span></h2>'
            '<div class="taxo">' + "".join(grid) + '</div>' + note + '</section>')


# ------------------------------------------------------------- C. RPG-RT curve SVG
_CURVE_ORDER = ["raw", "esd_u", "fcf_p_official", "odace_v3",
                "lsse_geo_e2", "odace_benign", "odace_benign_n1", "sph_ot"]
_CURVE_COL = {"raw": "#c0c6cc", "esd_u": "#a8b0b8", "fcf_p_official": "#86b6e0",
              "odace_v3": "#e0a060", "lsse_geo_e2": "#73c7b0", "odace_benign": "#6ad08e",
              "odace_benign_n1": "#4fbf87", "sph_ot": "#c090e0"}
_CURVE_NAME = {"raw": "Raw SD", "esd_u": "ESD-u", "fcf_p_official": "FCF-P", "odace_v3": "ODACE v3",
               "lsse_geo_e2": "LSSE geo", "odace_benign": "ODACE benign", "odace_benign_n1": "ODACE benign-neg",
               "sph_ot": "SLERP-OT"}


def rpgrt_curve_section(ctx: Ctx, dpo: dict) -> str:
    models = (dpo or {}).get("models", {}) or {}
    series, maxit, maxy = [], 1, 10.0
    for k in _CURVE_ORDER:
        cur = (models.get(k, {}) or {}).get("curve") or []
        seq = [(int(p["iter"]), float(p["asr_query"])) for p in cur if p.get("asr_query") is not None]
        if not seq:
            continue
        maxit = max(maxit, max(i for i, _ in seq))
        maxy = max(maxy, max(v for _, v in seq))
        series.append((k, seq))
    if not series:
        return ""
    W, H, PL, PR, PT, PB = 900, 420, 50, 158, 26, 46
    ymax = int(math.ceil(maxy / 10) * 10)
    px = lambda i: PL + (i / maxit) * (W - PL - PR)
    py = lambda v: H - PB - (v / ymax) * (H - PT - PB)
    s = ['<text x="{0}" y="15" text-anchor="middle" fill="#c8d0da" font-size="12" '
         'font-weight="600">Adaptive attacker strength over DPO iterations</text>'.format((PL + W - PR) / 2)]
    for gy in range(0, ymax + 1, 10):
        s.append('<line x1="{0}" y1="{1:.1f}" x2="{2}" y2="{1:.1f}" stroke="#343a4244"/>'.format(PL, py(gy), W - PR))
        s.append('<text x="{0}" y="{1:.1f}" text-anchor="end" fill="#a8b0b8" font-size="10">{2}</text>'.format(PL - 5, py(gy) + 3, gy))
    for gi in range(0, maxit + 1):
        s.append('<text x="{0:.1f}" y="{1}" text-anchor="middle" fill="#a8b0b8" font-size="10">{2}</text>'.format(px(gi), H - PB + 15, gi))
    s.append('<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#343a42"/>'.format(PL, PT, H - PB))
    s.append('<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#343a42"/>'.format(PL, H - PB, W - PR))
    s.append('<text x="{0:.1f}" y="{1}" text-anchor="middle" fill="#a8b0b8" font-size="11">DPO iteration '
             '(0 = frozen attacker)</text>'.format((PL + W - PR) / 2, H - PB + 32))
    s.append('<text x="14" y="{0}" text-anchor="middle" fill="#a8b0b8" font-size="11" '
             'transform="rotate(-90 14 {0})">RPG-RT ASR (% queries NSFW) &#8595;</text>'.format(H / 2))
    for k, seq in series:
        col = _CURVE_COL.get(k, "#888")
        pline = " ".join("{0:.1f},{1:.1f}".format(px(i), py(v)) for i, v in seq)
        s.append('<polyline points="{0}" fill="none" stroke="{1}" stroke-width="2.2" opacity="0.92"/>'.format(pline, col))
        for i, v in seq:
            s.append('<circle cx="{0:.1f}" cy="{1:.1f}" r="3" fill="{2}"><title>{3} iter{4}: {5:.1f}%</title>'
                     '</circle>'.format(px(i), py(v), col, _CURVE_NAME.get(k, k), i, v))
        li, lv = seq[-1]
        s.append('<text x="{0:.1f}" y="{1:.1f}" fill="{2}" font-size="10">{3}</text>'.format(
            W - PR + 6, py(lv) + 3, col, _CURVE_NAME.get(k, k)))
    svg = ('<svg viewBox="0 0 {0} {1}" xmlns="http://www.w3.org/2000/svg" '
           'style="background:var(--panel);border-radius:8px">{2}</svg>'.format(W, H, "".join(s)))
    note = ('<div class="legend"><b>True worst-case robustness.</b> The RPG-RT attacker DPO-fine-tunes '
            'itself on each target&rsquo;s feedback, so a rising curve means the defense is being learned. '
            'Redirect defenses (<span style="color:#c090e0">SLERP-OT</span>, '
            '<span style="color:#6ad08e">ODACE benign</span>) stay flat and low; the raw model climbs '
            '52.5&rarr;75. iter0 is the frozen best-of-N attacker. Our run: Vicuna-7B 4bit on one RTX&nbsp;4070.</div>')
    return ('<section class="sec"><h2>RPG-RT adaptation curve '
            '<span>(asr_query per DPO iteration &middot; flat &amp; low = adaptation-proof)</span></h2>'
            '<div class="pareto-wrap">' + svg + '</div>' + note + '</section>')


# --------------------------------------------------------- F. cross-concept transfer
_TRANSFER_ROWS = [
    ("raw_v14", "&mdash;", "&mdash;"),
    ("fcf_p_official", "nudity", "TE fine-tune"),
    ("esd_u", "nudity", "UNet non-Xattn"),
    ("odace_v3", "nudity", "UNet x-attn (output)"),
    ("lsse_r2q_ab", "nudity", "TE read-out space"),
    ("lsse_r2q_violence", "violence", "TE read-out space"),
]


def transfer_heatmap_section(ctx: Ctx, M: dict, vd: dict) -> str:
    head = ['<th class="mh">Model</th>',
            '<th class="muted">Trained&nbsp;concept</th>',
            '<th class="muted">Edit&nbsp;space</th>',
            '<th>Nudity&nbsp;ASR&nbsp;4-lab&nbsp;<span class="ar">&darr;</span></th>',
            '<th>Violence&nbsp;Q16&nbsp;<span class="ar">&darr;</span></th>']
    body = []
    for key, concept, space in _TRANSFER_ROWS:
        d = M.get(key, {}) or {}
        label, group, base, mod = ctx.model_info(key)
        viol = (vd.get(key, {}) or {}).get("mean")
        if viol is None:
            viol = d.get("violence")
        tds = [ctx.mh(label, key, group, base, mod),
               '<td class="num muted">{0}</td>'.format(concept),
               '<td class="num muted" style="font-size:11px">{0}</td>'.format(space),
               ctx.asr_cell(d.get("fs_mean4"), bold=True),
               ctx.asr_cell(viol)]
        body.append('<tr>' + "".join(tds) + '</tr>')
    note = ('<div class="legend"><b>Erasure is not always concept-local.</b> Output-grounded UNet editing '
            '(<b>ODACE</b>) erases nudity while leaving violence Q16 near the raw level (&asymp;59) &mdash; a '
            'concept-<i>local</i> edit. Read-out-space text-encoder erasure (<b>LSSE</b>) suppresses a broad '
            '&ldquo;unsafe&rdquo; direction, so a nudity-trained model also lowers violence, and the '
            'violence-trained model (16.7 Q16) leaves nudity largely intact. Both columns are ASR (lower=safer, '
            'heat-colored); prior FCF/ESD/SLD papers do not report this cross-concept breakdown.</div>')
    return ('<section class="sec"><h2>Cross-concept transfer '
            '<span>(does erasing one concept move the other? &middot; concept-locality check)</span></h2>'
            '<div class="wrap">' + ctx.table(head, body) + '</div>' + note + '</section>')


# -------------------------------------------------- V. validation (triangulation + 8-lab decomp)
_VAL_KEYS = ["raw_v14", "fcf_p_official", "sph_ot", "odace_benign_n1", "odace_benign",
             "lsse_geo_e2", "odace_v3", "lsse_r2q_ab"]
_SURPLUS_SHORT = {"FEMALE_BREAST_COVERED": "breast-covered", "BUTTOCKS_COVERED": "buttocks-covered",
                  "FEMALE_GENITALIA_COVERED": "genitalia-covered", "BUTTOCKS_EXPOSED": "buttocks-EXPOSED"}


def validation_section(ctx: Ctx, M: dict, coh_tri: dict, decomp: dict) -> str:
    """Two evidence tables that de-risk the coherence story: (1) a CLIP-independent detector
    corroborates the collapse probe; (2) the strict 8-lab surplus of coherent models is clothed."""
    esc = ctx.html.escape
    # --- Table 1: triangulation (CLIP ring vs independent Haar face_rate) ---
    h1 = ['<th class="mh">Model</th>',
          '<th>CLIP&nbsp;ring&nbsp;<span class="ar">&uarr;</span><br><span class="sub">person_prob</span></th>',
          '<th>face&nbsp;ring&nbsp;<span class="ar">&uarr;</span><br><span class="sub">Haar, CLIP-free</span></th>',
          '<th class="muted">face&nbsp;i2p<br><span class="sub">control</span></th>',
          '<th class="muted">agree?</th>']
    b1 = []
    for k in _VAL_KEYS:
        t = coh_tri.get(k)
        if not t:
            continue
        clip = (M.get(k, {}) or {}).get("coh_ring")
        fr = (t.get("ring_a_bell") or {}).get("face_rate")
        fi = (t.get("i2p") or {}).get("face_rate")
        if clip is None or fr is None:
            agree = "&#8212;"
        elif (clip < 0.4) == (fr < 0.3):
            agree = ('<span style="color:#6c6">yes &middot; coherent</span>' if clip >= 0.4
                     else '<span style="color:#e66">yes &middot; collapse</span>')
        else:
            agree = 'partial'
        b1.append('<tr><th class="mh">{0}</th>{1}{2}{3}<td class="num muted">{4}</td></tr>'.format(
            esc(ctx.labels.get(k, k)), ctx.ret_cell(clip), ctx.ret_cell(fr), ctx.ret_cell(fi), agree))
    # --- Table 2: 8-lab decomposition (exposed vs covered surplus) ---
    h2 = ['<th class="mh">Model</th>',
          '<th>ASR&nbsp;4-lab&nbsp;<span class="ar">&darr;</span><br><span class="sub">exposed</span></th>',
          '<th class="muted">ASR&nbsp;8-lab<br><span class="sub">strict</span></th>',
          '<th>8&minus;4&nbsp;surplus<br><span class="sub">covered-only</span></th>',
          '<th class="muted">dominant surplus label</th>']
    b2 = []
    for k in _VAL_KEYS:
        d = decomp.get(k)
        if not d or not d.get("n"):
            continue
        pl = d.get("per_label_rate", {})
        surplus = {lab: pl.get(lab, 0) for lab in _SURPLUS_SHORT}
        top = sorted(surplus.items(), key=lambda x: -x[1])
        tops = ", ".join("{0}&nbsp;{1:.0f}".format(_SURPLUS_SHORT[l], v) for l, v in top[:2] if v > 0) or "&#8212;"
        b2.append('<tr><th class="mh">{0}</th>{1}{2}{3}<td class="num muted" style="font-size:11px">{4}</td></tr>'.format(
            esc(ctx.labels.get(k, k)), ctx.asr_cell(d.get("asr_4lab"), bold=True),
            ctx.asr_cell(d.get("asr_8lab")), ctx.asr_cell(d.get("asr_covered_only")), tops))
    if not b1 and not b2:
        return ""
    note = ('<div class="legend"><b>Two independent de-risks of the coherence story.</b> '
            '<b>(1)</b> A CLIP-independent Haar face detector agrees with the CLIP probe '
            '(Pearson&nbsp;r&nbsp;=&nbsp;+0.68 over the curated set) &mdash; the collapse is not a CLIP '
            'artifact. <b>(2)</b> The coherent redirect models&rsquo; higher strict-8-lab is almost '
            'entirely <i>breast-covered</i> (clothed people), not exposed nudity: their 4-lab exposed '
            'ASR stays 1&ndash;4, so the low ASR is honest. See <code>coherence_tri.json</code> / '
            '<code>label_decomp.json</code>.</div>')
    return ('<section class="sec"><h2>Coherence validation '
            '<span>(independent corroboration &middot; 8-lab surplus decomposition)</span></h2>'
            '<div class="wrap">' + ctx.table(h1, b1) + '</div>'
            '<div class="wrap" style="margin-top:10px">' + ctx.table(h2, b2) + '</div>'
            + note + '</section>')


# --------------------------------------------------------------------- extra CSS
EXTRA_CSS = """
.hero{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:4px 0}
.hcard{background:linear-gradient(155deg,var(--panel2),var(--panel));border:1px solid var(--line);border-radius:10px;padding:14px 15px}
.hcard.accent{border-color:var(--accent);box-shadow:0 0 0 1px #5ec6a833 inset}
.hnum{font-size:25px;font-weight:800;color:var(--accent);line-height:1.05;font-variant-numeric:tabular-nums}
.hlbl{font-size:12px;font-weight:700;margin:6px 0 5px;letter-spacing:.01em}
.hdesc{font-size:11.5px;color:var(--muted);line-height:1.45}.hdesc b{color:var(--text)}
.taxo{display:grid;grid-template-columns:120px 1fr 1fr;gap:8px;margin:6px 0 4px}
.taxo .thd{font-size:12px;font-weight:700;color:var(--text);display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:6px;line-height:1.3}
.taxo .rhd{font-size:12px;font-weight:700;color:var(--text);display:flex;align-items:center;justify-content:center;writing-mode:vertical-rl;transform:rotate(180deg)}
.tcell{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px;min-height:84px}
.mchip{display:inline-block;font-size:11px;padding:3px 8px;margin:3px;border-radius:12px;border:1px solid var(--line);background:var(--panel2);color:var(--text)}
.mchip.coh{border-color:#3a7a5e;background:#14261e}
.mchip.col{border-color:#7a3a3a;background:#261414}
.mchip.par{border-color:#7a6a3a;background:#262214}
.mchip .muted{color:var(--muted)}
.oodhd{display:grid;grid-template-columns:190px 1fr 1fr;gap:12px;font-size:12px;font-weight:700;color:var(--muted);padding:10px 0 4px}
.oodrow{display:grid;grid-template-columns:190px 1fr 1fr;gap:12px;align-items:center;padding:9px 0;border-top:1px solid var(--line)}
.oodlab{font-size:12.5px;line-height:1.35}.oodlab .v-coh{color:#7cd0a4}.oodlab .v-col{color:#e08a8a}
.oodimgs{display:flex;gap:5px;flex-wrap:nowrap;overflow:hidden}
.oodimgs a{flex:1;min-width:0;max-width:82px}
.oodimgs img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:4px;background:#0b0c0d}
.badge{display:inline-block;font-size:10px;font-weight:700;padding:2px 7px;border-radius:10px}
.badge.coh{background:#173528;color:#7cd0a4}.badge.col{background:#351717;color:#e08a8a}
"""
