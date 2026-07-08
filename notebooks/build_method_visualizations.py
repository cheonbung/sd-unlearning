"""Generate `method_visualizations.ipynb` (Colab-ready) from stdlib json only.

Slide-ready explanatory figures for the CURRENT method story (2026-07):
ODACE benign-anchor/benign-neg redirect, Sph+OT geodesic + farthest-noise OT
endpoint selection, LSSE read-out-space geodesic erasure (`lsse_geo_e2`).
Pure matplotlib, consistent design system (serif + LaTeX mathtext, 220 dpi).

This file is auto-synced FROM the live notebook (2026-07-08): edit the
notebook, then regenerate this builder, or edit both. Running it rewrites
`method_visualizations.ipynb` with exactly the same cells.

Run:  python notebooks/build_method_visualizations.py
Out:  notebooks/method_visualizations.ipynb  (open in Colab, Runtime > Run all)
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "method_visualizations.ipynb")

CELLS = []
def md(src):   CELLS.append(("md", src))
def code(src): CELLS.append(("code", src))


md(r"""# Concept-Unlearning Methods - Explanatory Figures (HD)
This notebook regenerates slide-ready figures for the current method story.

Current framing:

| Method | Optimized object | Current variant to present | Core operation |
|---|---|---|---|
| **ODACE** | SD UNet cross-attention output/noise prediction | `benign_anchor` / `benign_neg` | redirect concept-prompt noise to a coherent benign target |
| **Sph+OT** | CLIP text encoder embedding geometry | spherical projection + optimal transport | move forbidden prompts on the CLIP sphere while preserving norm/rank structure |
| **LSSE geodesic** | CLIP text encoder under frozen-UNet read-out metric | `lsse_geo_e2` | rotate read-out embeddings away from the concept on-manifold, with retain anchoring |

**Run all** (`Runtime - Run all`). Each code cell renders one HD figure (220 dpi)
under `notebooks/figures/`, then packages them into `notebooks/method_figures.zip`.
""")

code(r"""# === Design system: palette, typography, depth helpers ======================
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D            # noqa: F401  (registers 3d)
from mpl_toolkits.mplot3d import proj3d

np.random.seed(42)
os.makedirs("figures", exist_ok=True)

# Curated, slightly desaturated palette (good on white, prints well).
COL = dict(
    ink="#1f2933", muted="#66727e", base="#aab4bf", grid="#e7ecef",
    prop="#0f8b7e",      # proposed (teal)
    forget="#d1495b",    # concept / removed (rose)
    safe="#4c9f70",      # safe / retained (green)
    axis="#3d4852",
    orth="#3b6ea5",      # orthogonal / null-space (blue)
    anchor="#7b62c4",    # retain/read-out anchor (violet)
    noise="#e0982f",     # OT noise (amber)
)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 220, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlecolor": COL["ink"],
    "axes.labelsize": 12, "axes.labelcolor": COL["ink"],
    "axes.edgecolor": "#cdd5dc", "axes.linewidth": 1.2,
    "xtick.color": COL["muted"], "ytick.color": COL["muted"],
    "text.color": COL["ink"],
    "legend.frameon": False, "legend.fontsize": 11,
})

def save(fig, name):
    p = f"figures/{name}.png"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
    print("saved", p)

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

def lbl(ax, xy, text, color=None, fc="white", ec=None, fs=12,
        ha="center", va="center", pad=0.34, alpha=0.97, z=11):
    color = color or COL["ink"]
    ax.annotate(text, xy, ha=ha, va=va, fontsize=fs, color=color, zorder=z,
                bbox=dict(boxstyle=f"round,pad={pad}", fc=fc,
                          ec=ec if ec is not None else color, lw=1.1, alpha=alpha))

def arrow(ax, p0, p1, color, lw=2.6, ls="-", z=6, ms=18):
    ax.annotate("", xy=p1, xytext=p0, zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, mutation_scale=ms))

def node(ax, p, color, s=95, z=8):
    ax.scatter(*p, s=s, color=color, zorder=z, edgecolors="white", linewidths=1.6)

def glow(line, lw=6, color="white", alpha=0.7):
    line.set_path_effects([pe.Stroke(linewidth=lw, foreground=color, alpha=alpha),
                           pe.Normal()])

class Arrow3D(FancyArrowPatch):
    # 3D arrow (standard recipe; works on modern matplotlib)
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs
    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min(zs))

def arrow3d(ax, p0, p1, color, lw=2.4, ls="-"):
    a = Arrow3D([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                mutation_scale=16, lw=lw, arrowstyle="-|>", color=color, ls=ls)
    ax.add_artist(a)

print("design system ready — matplotlib", plt.matplotlib.__version__)""")

code(r"""# === (Optional) Korean labels in Colab — uncomment to install a KR font ======
# !apt-get -qq install -y fonts-nanum > /dev/null
# import matplotlib.font_manager as fm
# fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
# plt.rcParams['font.family'] = 'NanumGothic'; plt.rcParams['axes.unicode_minus'] = False
# print("Korean font enabled")
pass""")

md(r"""## Overview — three mathematical spaces, three verbs""")

code(r"""# === Fig 00: three optimization spaces ======================================
fig, axes = plt.subplots(1, 3, figsize=(14, 4.7), constrained_layout=True)

# Panel 1: noise space (ODACE benign redirect)
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
ax.quiver(gx, gy, -gy, gx, color=COL["prop"], alpha=.45, width=.006, scale=24)
ax.set_title("ODACE", color=COL["prop"])
lbl(ax, (0, 1.04), r"$\epsilon\in\mathbb{R}^{C\times H\times W}$", fc="#eef6f4",
    ec=COL["prop"], fs=12)
ax.text(0, -1.34, "noise space -> redirect to benign", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)

# Panel 2: sphere with a geodesic (Sph+OT)
ax = axes[1]
t = np.linspace(0, 2*np.pi, 240)
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.8)
ax.fill(np.cos(t), np.sin(t), color=COL["grid"], alpha=.5, zorder=0)
th = np.linspace(.25, 1.5, 60)
g, = ax.plot(np.cos(th), np.sin(th), color=COL["prop"], lw=3.4); glow(g)
node(ax, (np.cos(.25), np.sin(.25)), COL["prop"])
node(ax, (np.cos(1.5), np.sin(1.5)), COL["forget"])
ax.set_title("Sph+OT", color=COL["prop"])
lbl(ax, (0, 1.16), r"$u\in\mathbb{S}^{d-1}$", fc="#eef6f4", ec=COL["prop"], fs=12)
ax.text(0, -1.4, "CLIP geometry -> move on sphere", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.45, 1.3)

# Panel 3: LSSE geodesic read-out space
ax = axes[2]
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.8)
ax.fill(np.cos(t), np.sin(t), color=COL["grid"], alpha=.5, zorder=0)
ang_z, ang_c = 1.35, 0.35
z0 = np.array([np.cos(ang_z), np.sin(ang_z)])
cc = np.array([np.cos(ang_c), np.sin(ang_c)])
arc_th = np.linspace(ang_z, ang_z + .55, 60)   # rotate away from the lower-angle concept point
arc = np.c_[np.cos(arc_th), np.sin(arc_th)]
g, = ax.plot(arc[:, 0], arc[:, 1], color=COL["prop"], lw=3.4); glow(g)
node(ax, z0, COL["axis"]); node(ax, cc, COL["forget"]); node(ax, arc[-1], COL["prop"])
ax.set_title("LSSE geodesic", color=COL["prop"])
lbl(ax, (0, 1.16), r"$R=C\,M_\ell^{1/2}$", fc="#eef6f4", ec=COL["prop"], fs=12)
ax.text(0, -1.4, "UNet read-out -> rotate on-manifold", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.45, 1.3)

for ax in axes:
    ax.set_aspect("equal"); clean(ax)
fig.suptitle("Unlearning re-framed: which space do we optimize in?",
             fontsize=16, fontweight="bold", color=COL["ink"])
save(fig, "00_overview_spaces")
plt.show()
""")

md(r"""## ODACE - output-grounded benign redirect in noise space

The earlier ODACE negative-guidance target

$$
\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)
$$

is not the final method to present. It lowered ASR in a narrow setting, but it produced OOD / Ring-A-Bell collapse because the target can move the denoising trajectory into an incoherent region.

The current ODACE story is output-grounded and benign-directed. The text encoder is frozen. Only UNet cross-attention projections are edited. For a concept prompt trajectory, the frozen teacher output for a benign prompt becomes the anchor:

$$
\epsilon_b = \epsilon_{\theta_0}(z_t,t,c_{benign}), \qquad
c_{benign}=\text{``a fully clothed person, photograph''}
$$

Two current variants:

$$
\textbf{benign-anchor:}\quad \epsilon_{target}=\epsilon_b
$$

$$
\textbf{benign-neg:}\quad \epsilon_{target}=\epsilon_b-\lambda(\epsilon_p-\epsilon_b)
$$

This means ODACE no longer says only "move away from the unsafe output". It says "make the concept prompt denoise like a coherent benign output, and optionally subtract the residual concept direction around that benign anchor".
""")

md(r"""### ODACE Notation And Components

ODACE freezes the CLIP text encoder and trains a small subset of the SD UNet: the cross-attention projections. The supervision is applied in the model's own denoising-output space, not in a text-proxy space.

| Symbol | Meaning |
|---|---|
| $z_t$ | latent at diffusion timestep $t$ |
| $\epsilon_\theta(z_t,t,c)$ | UNet noise prediction conditioned on prompt/text embedding $c$ |
| $\epsilon_0$ | frozen-teacher output for the empty (unconditional) prompt — shared reference on the same latent; appears only in the old push-away target |
| $\epsilon_p$ | frozen-teacher noise output for the forget/concept prompt |
| $\epsilon_b$ | frozen-teacher noise output for the benign prompt, e.g. ``a fully clothed person, photograph'' |
| $\epsilon_p-\epsilon_b$ | residual concept direction around the benign output |
| $\lambda$ | benign-neg strength; $\lambda=1$ subtracts the concept residual once around the benign anchor |

| Loss | Formula | Role |
|---|---|---|
| $L_{forget}$ | $\lVert\epsilon_\theta(z_t,t,c_{forget})-\epsilon_{target}\rVert^2$ | redirect concept prompts to a coherent benign output |
| $L_{retain}$ | $\lVert\epsilon_\theta(z_t,t,c_{retain})-\epsilon_{\theta_0}(z_t,t,c_{retain})\rVert^2$ | keep non-target prompts close to the frozen teacher |
| $L_{total}$ | $\alpha L_{forget}+\beta L_{retain}$ | balance erasure and locality |

**Variants to mention**

| Variant | Trainable module | Status |
|---|---|---|
| ODACE **benign-anchor** | full cross-attention $W_{Q,K,V,O}$ | coherent non-collapsed redirect to benign output |
| ODACE **benign-neg** | full cross-attention $W_{Q,K,V,O}$ | strongest coherent ODACE point so far |
| ODACE **v3/v1.5 negative-guidance** | full cross-attention, old push-away target | excluded because Ring-A-Bell / OOD generation collapsed |
| ODACE **v2 K/V-only** | cross-attention K/V only | excluded because erasure was too weak |
""")

md(r"""### ODACE Intuition - Why Benign Redirect Replaced Push-Away

**Q1. What changed from the older ODACE slide?**
The older slide used a push-away target, $\epsilon_0-\eta(\epsilon_p-\epsilon_0)$. That target can leave the teacher's coherent denoising manifold. The final story uses the frozen UNet itself to define a valid benign target on the same latent/timestep trajectory.

**Q2. Why is this still different from text-encoder methods?**
FCF, LSSE, DACE, and Sph+OT mainly edit or move CLIP text embeddings while the UNet is frozen. ODACE edits the UNet cross-attention projections directly and supervises the actual denoising output. ESD is a closer UNet-edit baseline, but ODACE's final target is benign-output redirection rather than pure negative guidance.

| Target | Equation | Interpretation | Outcome |
|---|---|---|---|
| push-away | $\epsilon_0-\eta(\epsilon_p-\epsilon_0)$ | move away from the concept output | low ASR can hide OOD collapse |
| benign-anchor | $\epsilon_b$ | map concept prompt to a clothed-person output | coherent |
| benign-neg | $\epsilon_b-\lambda(\epsilon_p-\epsilon_b)$ | anchor at benign output and subtract concept residual | best coherent ODACE point |

**Q3. What should the seminar emphasize?**
Do not present ODACE as "stronger negative guidance". Present it as **teacher-output redirection to a benign anchor**. The important evaluation point is that ASR must be read together with Ring-A-Bell coherence/person-presence; otherwise a collapsed image can look successful to a detector.
""")

code(r"""# === Fig O1: benign-redirect vector decomposition ==========================
fig, ax = plt.subplots(figsize=(8.0, 6.3))
e0 = np.array([0.0, 0.0])
ep = np.array([1.75, 1.05])
eb = np.array([-0.95, 0.62])
lam = 0.65
etarget = eb - lam * (ep - eb)

# Reference arrows from the same frozen latent.
arrow(ax, tuple(e0), tuple(ep), COL["forget"], lw=3)
arrow(ax, tuple(e0), tuple(eb), COL["safe"], lw=3)
arrow(ax, tuple(eb), tuple(etarget), COL["prop"], lw=3)
ax.plot([ep[0], eb[0]], [ep[1], eb[1]], color="#c3c9cf", lw=1.5, ls=(0, (4, 3)))
node(ax, e0, COL["axis"]); node(ax, ep, COL["forget"]); node(ax, eb, COL["safe"]); node(ax, etarget, COL["prop"])

lbl(ax, (-0.28, -0.22), r"$\epsilon_0$", color=COL["axis"], fs=13)
lbl(ax, (ep[0]+0.10, ep[1]+0.14), r"$\epsilon_p$  concept output",
    color=COL["forget"], fc="#fdeef0", fs=12)
lbl(ax, (eb[0]-0.22, eb[1]+0.22), r"$\epsilon_b$  benign output",
    color=COL["safe"], fc="#eef8f0", fs=12)
lbl(ax, ((ep[0]+eb[0])/2+0.05, (ep[1]+eb[1])/2-0.25),
    r"$\epsilon_p-\epsilon_b$", color=COL["muted"], fs=11)
lbl(ax, (etarget[0]-0.12, etarget[1]-0.28),
    r"$\epsilon_{target}=\epsilon_b-\lambda(\epsilon_p-\epsilon_b)$",
    color=COL["prop"], fc="#eaf5f2", fs=12)

ax.set_title("ODACE benign-neg redirects the output through a coherent target")
ax.set_xlim(-2.85, 2.25); ax.set_ylim(-1.0, 1.75); ax.set_aspect("equal"); clean(ax)
save(fig, "O1_benign_redirect_vectors")
plt.show()
""")

code(r"""# === Fig O2: trajectory redirect vs push-away collapse =====================
fig, ax = plt.subplots(figsize=(8.2, 6.3))
forget = np.array([1.55, -1.25])
benign = np.array([-1.45, -1.05])
collapse = np.array([0.15, -1.85])
start = np.array([0.0, 1.85])

def bezier(p0, p1, p2, n=90):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2

base = bezier(start, np.array([0.95, 0.25]), forget)
push = bezier(start, np.array([0.05, -0.05]), collapse)
redir = bezier(start, np.array([-0.95, 0.25]), benign)

for c, col, lab, r in [
    (forget, COL["forget"], "concept mode", 0.58),
    (benign, COL["safe"], "benign coherent mode", 0.58),
    (collapse, COL["base"], "collapse / no person", 0.45),
]:
    ax.add_patch(plt.Circle(tuple(c), r, color=col, alpha=.14, zorder=0))
    node(ax, c, col)
    ax.text(c[0], c[1]-0.88, lab, ha="center", color=col, fontsize=11)

lb, = ax.plot(base[:, 0], base[:, 1], color=COL["base"], lw=3.1, label="raw trajectory"); glow(lb)
lp, = ax.plot(push[:, 0], push[:, 1], color=COL["forget"], lw=2.8, label="old push-away"); glow(lp)
lr, = ax.plot(redir[:, 0], redir[:, 1], color=COL["prop"], lw=3.4, label="benign redirect"); glow(lr)
for curve, col in [(base, COL["base"]), (push, COL["forget"]), (redir, COL["prop"] )]:
    arrow(ax, tuple(curve[-9]), tuple(curve[-1]), col, lw=3.0)
node(ax, start, COL["axis"]); lbl(ax, (start[0]+0.30, start[1]+0.04), r"$z_T$", fs=12)

ax.set_title("Benign redirect preserves coherence while avoiding the concept")
ax.legend(loc="upper right")
ax.set_xlim(-2.45, 2.45); ax.set_ylim(-2.55, 2.35); ax.set_aspect("equal"); clean(ax)
save(fig, "O2_benign_redirect_trajectory")
plt.show()
""")

code(r"""# === Fig O3: conventional text-proxy  vs  ODACE benign output supervision ===
def card(ax, x, y, w, h, text, fc, ec, fs=11, tc=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018",
                 fc=fc, ec=ec, lw=1.8, mutation_aspect=1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=tc or COL["ink"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

ax = axes[0]; ax.set_title("Text-proxy methods -> edit CLIP embedding", color=COL["base"])
card(ax, 0.04, 0.55, 0.24, 0.2, "prompt $c$", "#eef1f4", "#aab4bf")
card(ax, 0.38, 0.5, 0.26, 0.3, "edit $T_\\theta$\n(text space)", "#fdeef0", COL["forget"],
     tc=COL["forget"])
card(ax, 0.74, 0.5, 0.22, 0.3, "UNet\n(frozen)", "#eef1f6", "#9aa7c4")
arrow(ax, (0.28, 0.65), (0.38, 0.65), COL["muted"], lw=2)
arrow(ax, (0.64, 0.65), (0.74, 0.65), COL["muted"], lw=2)
ax.annotate("", xy=(0.85, 0.46), xytext=(0.85, 0.34),
            arrowprops=dict(arrowstyle="-|>", color="#c3c9cf", lw=1.6))
lbl(ax, (0.5, 0.24), "image effect is indirect -> leakage risk", color=COL["forget"],
    fc="#fdeef0", fs=11)

ax = axes[1]; ax.set_title("ODACE -> supervise BENIGN output $\\epsilon$", color=COL["prop"])
card(ax, 0.04, 0.55, 0.24, 0.2, "prompt $c_{forget}$", "#eef1f4", "#aab4bf", fs=10)
card(ax, 0.38, 0.5, 0.26, 0.3, "UNet cross-attn\n$W_{Q,K,V,O}$", "#eaf5f2", COL["prop"],
     tc=COL["prop"])
card(ax, 0.74, 0.5, 0.22, 0.3, r"$\epsilon_\theta$", "#eaf5f2", COL["prop"], fs=16,
     tc=COL["prop"])
arrow(ax, (0.28, 0.65), (0.38, 0.65), COL["muted"], lw=2)
arrow(ax, (0.64, 0.65), (0.74, 0.65), COL["muted"], lw=2)
ax.annotate("", xy=(0.85, 0.46), xytext=(0.85, 0.34),
            arrowprops=dict(arrowstyle="-|>", color=COL["prop"], lw=2))
lbl(ax, (0.85, 0.26), r"$\|\epsilon_\theta-\epsilon_b\|^2$ or benign-neg",
    color=COL["prop"], fc="#eaf5f2", fs=11)
ax.text(0.5, 0.1, "DIRECT supervision on coherent generated noise", ha="center",
        color=COL["prop"], fontsize=11)

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0.03, 0.92); ax.axis("off")
save(fig, "O3_proxy_vs_output")
plt.show()
""")

code(r"""# === Fig O4: locality - only cross-attention is trained =====================
fig, ax = plt.subplots(figsize=(10, 4.4))
blocks = ["Conv in", "ResBlock", "Self-Attn", "Cross-Attn", "ResBlock", "Conv out"]
trained = [False, False, False, True, False, False]
x = np.linspace(0.04, 0.82, len(blocks)); w = 0.12
for xi, b, tr in zip(x, blocks, trained):
    fc, ec = ("#eaf5f2", COL["prop"]) if tr else ("#eef1f4", "#bcc4cc")
    ax.add_patch(mpatches.FancyBboxPatch((xi, 0.42), w, 0.24, boxstyle="round,pad=0.012",
                 fc=fc, ec=ec, lw=2.2 if tr else 1.3))
    ax.text(xi+w/2, 0.54, b, ha="center", va="center", fontsize=9.5,
            color=ec if tr else COL["muted"])
    if tr:
        lbl(ax, (xi+w/2, 0.27), r"train $W_Q,W_K,W_V,W_O$", color=COL["prop"],
            fc="#eaf5f2", fs=10)
        ax.annotate("", xy=(xi+w/2, 0.40), xytext=(xi+w/2, 0.33),
                    arrowprops=dict(arrowstyle="-|>", color=COL["prop"], lw=1.8))
for i in range(len(x)-1):
    arrow(ax, (x[i]+w, 0.54), (x[i+1], 0.54), "#c3c9cf", lw=1.6)
ax.text(0.46, 0.8, "UNet: frozen except cross-attention", ha="center", fontsize=14,
        fontweight="bold", color=COL["ink"])
ax.text(0.46, 0.1, "small trainable footprint -> redirect concept prompts, keep coherent images",
        ha="center", color=COL["muted"], fontsize=10.5)
ax.set_xlim(0, 0.98); ax.set_ylim(0.03, 0.9); ax.axis("off")
save(fig, "O4_crossattn_locality")
plt.show()
""")

md(r"""## Sph+OT — Riemannian geometry + optimal transport
$$u_{new}=\mathrm{Exp}_{u_x}\!\big(-\mu_p\,\mathrm{Log}_{u_x}(u_c)\big),\qquad
\text{score}=\mathcal{W}(Z_{noise},Z_{explicit}).$$""")

md(r"""### 📑 기호 표 (Notation)

아래 그림(S1·S1b·S2·S3)에서 쓰는 표기와 동일합니다.

**구면 기하 (Sph)**

| 기호 | 명칭 | 의미 |
|---|---|---|
| $x$ | text embedding | 원본 텍스트 임베딩 벡터 ($\in\mathbb{R}^{d}$) |
| $d$ | dimension | 임베딩 차원 |
| $\lVert\cdot\rVert$ | L2 norm | 벡터 크기(길이) |
| $u_x=x/\lVert x\rVert$ | unit embedding | 정규화된 임베딩 — 구면 $\mathbb{S}^{d-1}$ 위의 점 |
| $u_c$ | concept direction | 지울 개념의 단위 방향(forget 축) |
| $\langle\cdot,\cdot\rangle$ | inner product | 내적(코사인 유사도의 분자) |
| $\theta=\arccos\langle u_x,u_c\rangle$ | geodesic angle | 두 점 사이 각(= 측지 거리) |
| $\mathbb{S}^{d-1}$ | unit hypersphere | 반지름 1의 초구(unit sphere) |
| $T_p\mathbb{S}$ | tangent space | 점 $p$에서의 접평면 |
| $v$ | tangent vector | 접벡터(이동 방향·속도) |
| $\mathrm{Log}_p(q)$ | log map | $p$에서 $q$로 향하는 접벡터로 변환 |
| $\mathrm{Exp}_p(v)$ | exp map | $p$에서 측지선 따라 실제로 이동 |
| $\mu_p$ | erase strength | 삭제 강도 = 멀어지는 각의 비율(스텝 크기) |
| $u_{new}$ | erased embedding | 개념을 지운 결과(여전히 구면 위) |

**최적수송 (OT)**

| 기호 | 명칭 | 의미 |
|---|---|---|
| $\mathcal{W}$ | (sliced-)Wasserstein | 두 분포 간 최적수송 거리 |
| $Z_{noise}$ | candidate dist. | 후보 노이즈가 만드는 출력 분포 |
| $Z_{explicit}$ | concept dist. | 개념이 명시된 기준 분포 |
| $z^{\star}=\arg\max_{z_i}\mathcal{W}$ | selected noise (farthest) | $\mathcal{W}$를 최대화하는 노이즈 = 개념 분포에서 **가장 먼**(개념과 가장 무관한) 노이즈 → forget endpoint 어휘로 사용 (코드: 후보 500개 중 상위 100개 선택) |
| $\omega$ | projection dir. | 1D 사영(슬라이싱) 방향 |
| $F^{-1}$ | quantile func. | 역누적분포(1D $W_2$ 닫힌 형에 사용) |

**대조 — FCF (기존 연구)**

| 기호 / 수식 | 의미 |
|---|---|
| $\mathrm{proj}_c(x)=\langle x,u_c\rangle\,u_c$ | 개념 축으로의 사영 성분 |
| $x_{clean}=\dfrac{x-\eta\,\mathrm{proj}_c(x)}{1-\eta}$ | FCF 정제식 (Euclidean, **현**을 따라감) |
| $\eta$ | FCF 사영 제거 강도 — upstream 코드명 `eta_clean`, 논문 표기 $\mu_p{=}0.7$ |

> $\mu_p$(Sph: 측지 *각도* 비율)와 $\eta$(FCF: 직선 *보간* 계수)는 이름은 비슷하지만 **역할이 다릅니다** — 같은 '삭제 강도'라도 한쪽은 각도를, 한쪽은 직선 거리를 조절합니다. 그림 S1·S1b·S2의 $x-\mu c$는 직선-빼기(chord) 대비를 위한 단순화 도식이며, FCF-P의 실제 식은 사영 성분을 뺀 뒤 $1/(1-\eta)$로 재스케일합니다.""")

md(r"""### 💡 직관 보강 — 자주 나오는 의문

**Q1. 왜 하필 '단위구(unit sphere)'인가?**
CLIP은 코사인 유사도 $\cos(x,c)=\langle x,c\rangle/(\lVert x\rVert\,\lVert c\rVert)$ 로 학습됩니다 — **크기는 무시하고 방향만** 비교하죠. 그래서 의미는 벡터의 길이가 아니라 *방향*에 담기고, 모든 임베딩은 사실상 **반지름이 고정된 구면 위의 점**으로 취급됩니다. "개념을 지운다 = 구면 위에서 그 방향으로부터 멀어진다." 반지름은 어차피 임의이므로 **1로 고정**(= 단위구)하는 것이 표준 관례입니다.

**Q2. '현(chord)을 따라간다'는 게 — 무엇이, 언제?**
개념 성분을 빼는 편집 스텝에서 **텍스트 임베딩 점 그 자체**가 움직입니다. FCF의 $x-\eta\,\mathrm{proj}_c(x)$ 는 삭제 강도를 키울수록 점을 **직선으로** 끌고 가는데, 이 직선이 구 내부를 가로지르는 **지름길(현/할선)** 입니다. → 끝점이 구 **안쪽**으로 떨어져 **노름이 줄어듦**(그림 S2). 측지선은 표면을 따라 도는 **호(arc)** 라서 끝점이 구 위에 그대로 남습니다(그림 S1).

| | 경로 | 끝점 노름 | 위치 |
|---|---|---|---|
| FCF (직선 빼기) | 현 / 할선 | 줄어듦 (<1) | 구 **안쪽** ✗ |
| Sph+OT (측지선) | 호(arc) | 보존 (=1) | 구 **표면** ✓ |

*2D 예시* ($u_x$ 각 $60°$, 개념축 $x$, 강도 $0.5$): FCF 빼기(보정 전) $\to(0.25,\,0.87)$ 노름 **0.90**(안쪽); 측지선은 $0.5\times60°=30°$ 회전 $\to(0.00,\,1.00)$ 노름 **1.00**(표면). 둘 다 개념 성분은 줄이지만 끝나는 위치가 다릅니다.

**Q3. 그럼 FCF의 $/(1-\eta)$ 는 무엇인가?**
구 안으로 빠진 점을 **반지름 방향으로 다시 늘려** 구 쪽으로 밀어주는 보정입니다. 하지만 반지름 스케일링 $\neq$ 측지선 회전이라, 보정 후에도 측지선이 닿는 **정확한 표면 점과는 다른 위치**에 떨어집니다 — '현으로 간 뒤 억지로 표면에 붙이기' vs '처음부터 표면을 따라가기'의 차이입니다.

**Q4. OT는 왜 필요한가?**
FCF는 forget endpoint로 **임의의** 노이즈 프롬프트를 쓰기 때문에, 무작위 노이즈가 우연히 개념 분포 *근처*에 놓이면 그쪽으로 옮겨도 개념이 제대로 지워지지 않습니다. OT는 각 후보 노이즈의 임베딩 분포와 개념 분포 사이 Wasserstein 거리를 계산해, **개념에서 가장 먼 노이즈들**($z^{\star}=\arg\max\mathcal{W}$, 코드는 500개 중 상위 100개)만 endpoint 어휘로 남깁니다 — "어디로 지울지"를 보장하는 endpoint 선택 장치입니다. 실제로 이 OT 선택이 적응형 공격 Ring-A-Bell(Re) 강건성을 크게 개선했습니다(그림 S3).""")

code(r"""# === Sphere helpers (Log / Exp / geodesic on the unit sphere) ===============
def log_map(p, q):
    p = p/np.linalg.norm(p); q = q/np.linalg.norm(q)
    d = np.clip(np.dot(p, q), -1, 1); theta = np.arccos(d)
    if theta < 1e-9:
        return np.zeros_like(p)
    u = q - d*p; u = u/np.linalg.norm(u)
    return theta*u

def exp_map(p, v):
    n = np.linalg.norm(v)
    return p if n < 1e-9 else np.cos(n)*p + np.sin(n)*(v/n)

def geodesic(p, q, ts):
    v = log_map(p, q)
    return np.array([exp_map(p, t*v) for t in ts])

print("sphere ops ready")""")

code(r"""# === Fig S1: geodesic step (Sph) vs Euclidean subtraction (3D) ==============
fig = plt.figure(figsize=(8.6, 7.6))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()

u = np.linspace(0, 2*np.pi, 60); v = np.linspace(0, np.pi, 30)
xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
zs = np.outer(np.ones_like(u), np.cos(v))
ax.plot_surface(xs, ys, zs, color="#d7e9e4", alpha=0.15, linewidth=0, shade=True)
ax.plot_wireframe(xs, ys, zs, color="#e6ebed", linewidth=0.3, rstride=4, cstride=4)

u_x = np.array([-0.15, -0.55, 0.82]); u_x /= np.linalg.norm(u_x)
u_c = np.array([0.95, 0.10, 0.20]); u_c /= np.linalg.norm(u_c)
mu = 0.85; v_log = log_map(u_x, u_c); u_new = exp_map(u_x, -mu*v_log)
arc_ref = geodesic(u_x, u_c, np.linspace(0, 1, 60))
arc_mov = np.array([exp_map(u_x, -t*mu*v_log) for t in np.linspace(0, 1, 60)])
eucl = u_x - mu*u_c

ax.plot(*arc_ref.T, color="#b3bcc1", lw=1.6, ls=(0, (4, 3)))   # ref geodesic -> u_c
ax.plot(*arc_mov.T, color=COL["prop"], lw=4)                    # the geodesic move
ax.plot([0, u_x[0]], [0, u_x[1]], [0, u_x[2]], color="#d2d9dd", lw=1.0)
ax.plot([0, eucl[0]], [0, eucl[1]], [0, eucl[2]], color=COL["base"], lw=1.4)
ax.plot([eucl[0], u_new[0]], [eucl[1], u_new[1]], [eucl[2], u_new[2]],
        color=COL["base"], ls=":", lw=1.6)                      # off-sphere gap
for p, c in [(u_x, COL["axis"]), (u_c, COL["forget"]), (u_new, COL["prop"]),
             (eucl, COL["base"])]:
    ax.scatter(*p, color=c, s=55, edgecolors="white", linewidths=1.4, depthshade=False)

def tlabel(p, txt, c, off):
    q = np.asarray(p) + np.array(off)
    ax.text(q[0], q[1], q[2], txt, color=c, fontsize=12, fontweight="bold", ha="center",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec=c, lw=1.0, alpha=0.92))
tlabel(u_x,  r"$u_x$",                  COL["axis"],   (0.0, 0.0, 0.30))
tlabel(u_c,  r"$u_c$",                  COL["forget"], (0.30, 0.14, 0.0))
tlabel(u_new, r"$u_{new}$",             COL["prop"],   (-0.42, -0.05, 0.12))
tlabel(eucl, r"$x-\mu c$ (off-sphere)", COL["base"],   (0.0, 0.0, -0.36))

ax.set_title("Geodesic stays on the sphere;  Euclidean subtraction leaves it", pad=0)
ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=18, azim=40)
save(fig, "S1_geodesic_vs_euclidean")
plt.show()""")

code(r"""# === Fig S1b: FCF (Euclidean)  vs  Sph+OT (geodesic) — 2D contrast ==========
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), constrained_layout=True)
t = np.linspace(np.pi*0.05, np.pi*0.95, 200)
ang_x, ang_c = 1.15, 0.35
ux = np.array([np.cos(ang_x), np.sin(ang_x)]); uc = np.array([np.cos(ang_c), np.sin(ang_c)])
mu = 0.6

for ax, mode in [(axes[0], "fcf"), (axes[1], "sph")]:
    ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=2)            # unit arc
    ax.fill_between(np.cos(t), np.sin(t), 0, color=COL["grid"], alpha=.45, zorder=0)
    node(ax, ux, COL["axis"]); node(ax, uc, COL["forget"])
    lbl(ax, (ux[0]-0.02, ux[1]+0.14), r"$u_x$", color=COL["axis"], fs=12)
    lbl(ax, (uc[0]+0.12, uc[1]+0.08), r"$u_c$", color=COL["forget"], fs=12)
    if mode == "fcf":
        res = ux - mu*uc
        arrow(ax, tuple(ux), tuple(res), COL["base"], lw=3)
        node(ax, res, COL["base"])
        ax.plot([res[0], res[0]], [0, res[1]], color="#c3c9cf", ls=":", lw=1)
        lbl(ax, (res[0]-0.05, res[1]-0.16), r"$x-\mu c$", color=COL["base"], fs=12)
        ax.annotate("off the arc\n(norm distorted, OOD)", (res[0], res[1]),
                    xytext=(res[0]-0.15, res[1]-0.62), ha="center", color=COL["forget"],
                    fontsize=11, arrowprops=dict(arrowstyle="-|>", color=COL["forget"]))
        ax.set_title("FCF — Euclidean subtraction", color=COL["base"])
    else:
        v = log_map(ux, uc); un = exp_map(ux, -mu*v)
        arc = np.array([exp_map(ux, -s*mu*v) for s in np.linspace(0, 1, 40)])
        g, = ax.plot(arc[:, 0], arc[:, 1], color=COL["prop"], lw=3.4); glow(g)
        node(ax, un, COL["prop"])
        lbl(ax, (un[0]-0.04, un[1]-0.16), r"$u_{new}$", color=COL["prop"], fs=12)
        ax.annotate("stays on the arc\n(norm preserved)", (un[0], un[1]),
                    xytext=(un[0]-0.1, un[1]-0.62), ha="center", color=COL["prop"],
                    fontsize=11, arrowprops=dict(arrowstyle="-|>", color=COL["prop"]))
        ax.set_title("Sph+OT — geodesic step", color=COL["prop"])
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.15, 1.2); ax.set_aspect("equal"); clean(ax)
fig.suptitle("Same goal, different geometry: straight line leaves the sphere; "
             "the geodesic does not", fontsize=14, fontweight="bold", color=COL["ink"])
save(fig, "S1b_fcf_vs_sphot")
plt.show()""")

code(r"""# === Fig S2: norm distortion vs preserved norm =============================
fig, ax = plt.subplots(figsize=(7.8, 5.2))
mus = np.linspace(0, 1, 120)
ux = np.array([1.0, 0.0]); uc = np.array([np.cos(0.9), np.sin(0.9)])
ne = [np.linalg.norm(ux - m*uc) for m in mus]
l1, = ax.plot(mus, ne, color=COL["base"], lw=3.4, label=r"Euclidean $\|x-\mu c\|$"); glow(l1)
ax.axhline(1.0, color=COL["prop"], lw=3.4, label=r"Geodesic (norm $=\|x\|$, preserved)")
ax.fill_between(mus, ne, 1.0, where=np.array(ne) < 1, color=COL["forget"], alpha=.12)
lbl(ax, (0.62, 0.6), "distortion / OOD drift", color=COL["forget"], fc="#fdeef0", fs=11)
ax.set_xlabel(r"forget strength $\mu$"); ax.set_ylabel("embedding norm")
ax.set_title("Euclidean editing distorts the norm; the geodesic does not")
ax.legend(loc="lower left"); ax.set_xlim(0, 1); ax.set_ylim(0.25, 1.15)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=COL["grid"], lw=1)
save(fig, "S2_norm_distortion")
plt.show()""")

code(r"""# === Fig S3: OT — pick the noise that MAXIMIZES Wasserstein =================
def sliced_w(A, B, n_proj=100):
    A = np.asarray(A); B = np.asarray(B); d = A.shape[1]; tot = 0.0
    q = np.linspace(0, 1, 100)
    for _ in range(n_proj):
        th = np.random.randn(d); th /= np.linalg.norm(th)
        tot += np.mean(np.abs(np.quantile(A @ th, q) - np.quantile(B @ th, q)))
    return tot / n_proj

explicit = np.random.randn(220, 2) * 0.45
cands = {"A": np.array([0.7, 0.6]), "B": np.array([-1.8, 1.6]),
         "C": np.array([2.4, -2.0]), "D": np.array([-0.4, -1.0])}
clouds = {k: np.random.randn(130, 2)*0.34 + c for k, c in cands.items()}
scores = {k: sliced_w(clouds[k], explicit) for k in clouds}
best = max(scores, key=scores.get)

fig, ax = plt.subplots(figsize=(8, 6.6))
ax.scatter(*explicit.T, s=16, color=COL["forget"], alpha=.45,
           label=r"$Z_{explicit}$ (concept)")
for k, pts in clouds.items():
    b = (k == best); c = COL["noise"] if b else COL["base"]
    ax.scatter(*pts.T, s=16, alpha=.75, color=c)
    cx, cy = cands[k]
    lbl(ax, (cx, cy+0.55), f"cand {k}\n$\\mathcal{{W}}$={scores[k]:.2f}",
        color=c if b else COL["muted"], fc="#fff7ea" if b else "white",
        ec=c, fs=11 if b else 10)
    if b:
        arrow(ax, tuple(explicit.mean(0)), (cx, cy), COL["noise"], lw=2.6)
ax.set_title("OT noise selection — keep the candidate with MAX $\\mathcal{W}$")
ax.text(0.5, -0.04, "farthest from $Z_{explicit}$ = safest forget endpoint (code keeps top-100/500)",
        transform=ax.transAxes, ha="center", fontsize=10.5, color=COL["muted"])
ax.legend(loc="upper left"); ax.set_aspect("equal"); clean(ax)
save(fig, "S3_optimal_transport_noise")
plt.show()""")

md(r"""## LSSE Geodesic - Read-Out-Space Geodesic Erasure + Retain Anchor

The final LSSE model to present is `lsse_geo_e2`, not the older "CNP + PLU + W2" slide. The key change is that the erasure target is defined in the frozen SD UNet's cross-attention read-out space.

For each relevant cross-attention layer $\ell$, build a frozen read-out metric:

$$
M_\ell \approx W_{k,\ell}^T W_{k,\ell}+W_{v,\ell}^T W_{v,\ell}, \qquad
R_\ell = C M_\ell^{1/2}
$$

Then LSSE moves the frozen read-out embedding away from the concept direction by a spherical/geodesic step, and trains the text encoder to match that target:

$$
R_\ell^{target}=\operatorname{GeoAway}(R_{\ell,0}, c_{read}, \eta)
$$

$$
L = \alpha L_{geo} + \beta L_{retain} + \beta L_{anchor}\;(+\,\gamma L_{implicit})
$$

PLU is a layer-unlock *schedule* (not a loss term), and the OOD list augments the forget prompts.

Seminar shorthand: **"edit the text encoder, but measure and redirect it through the frozen UNet read-out metric."**
""")

md(r"""### LSSE Geodesic Components

`LSSE geodesic` edits only the CLIP text encoder. The UNet and VAE remain frozen. The important distinction from the old slide is that the concept operation is no longer explained as a simple raw-CLIP null-space projection. It is a geodesic redirect measured through how the frozen UNet reads text embeddings.

| Component | Config / object | What it does | Why it matters |
|---|---|---|---|
| Read-out metric | `use_cap_cnp`, $M_\ell^{1/2}$ | maps CLIP embeddings to $R_\ell=C M_\ell^{1/2}$ | aligns the text edit with UNet cross-attention sensitivity |
| Concept direction | `contrastive_ortho` | estimates a concept direction while separating retain directions | avoids treating every nearby retain prompt as the concept |
| Geodesic loss | `cap_loss_mode="geodesic"` | builds an on-sphere target away from the concept mean | reduces off-manifold push-away behavior |
| Per-layer causal metric | `perlayer_causal` | weights layers by concept-causality | focuses the loss where the UNet reacts to the concept |
| Read-out retain anchor | `cap_retain_anchor=True` | keeps retain prompts close under the same read-out metric | protects locality while erasing |
| PLU | `use_plu=True` | unlocks trainable TE layers 1 → 3 → 6 at 1/3·2/3 of training (early / CAP-causal layers first) | limits sudden drift during training |
| OOD augmentation | `nudity_ood_aug.txt` | includes concept-near OOD prompts | checks against low-ASR collapse |

| Symbol | Meaning |
|---|---|
| $C$ | CLIP text-encoder token embedding |
| $M_\ell^{1/2}$ | square-root read-out metric from frozen UNet cross-attention layer $\ell$ |
| $R_\ell=C M_\ell^{1/2}$ | text embedding as seen by the UNet read-out |
| $c_{read}$ | concept direction / concept mean in read-out space |
| $\eta$ | geodesic step size; `lsse_geo_e2` uses `cap_redirect_strength=2.0` |
| $w_\ell$ | per-layer causal weight (forget loss only — the retain anchor averages layers uniformly) |

Do not delete PLU from the explanation. PLU remains part of the final model. What should be removed is the old framing that made `PLU + W2/null-space` sound like the main method. The main method is **geodesic read-out erasure with retain anchoring**.
""")

md(r"""### LSSE Geodesic Intuition

**Q1. Why not explain it only in raw CLIP space?**
Stable Diffusion does not consume a raw concept vector directly. Cross-attention reads text embeddings through its K/V projections. The read-out metric approximates this sensitivity, so the edit is optimized in the space that better matches image behavior.

**Q2. Why geodesic?**
A Euclidean push-away can increase norm or move embeddings into regions that do not correspond to natural prompts. The geodesic target rotates the frozen embedding away from the concept while keeping it on the same spherical geometry. That is the conceptual bridge from FCF's CLIP-sphere idea to the current LSSE model.

**Q3. What does the retain anchor add?**
A forget-only loss can erase by damaging nearby benign or retain prompts. The retain anchor uses the same read-out metric to keep retain prompts close to the frozen text encoder as seen by the UNet.

**Q4. How should the old LSSE+PLU+W2 slide change?**
Replace the title and diagram focus. PLU stays as a training stabilizer. W2/null-space should not be the headline. The headline should be: **read-out metric -> geodesic concept redirect -> retain anchor -> PLU/OOD robustness**.

**Q5. One-line seminar summary**
LSSE geodesic edits the text encoder, but the loss is defined by the frozen UNet's cross-attention read-out geometry, so concept removal is closer to what the image generator actually uses.
""")

code(r"""# === Fig L1: frozen UNet read-out metric ==================================
fig, ax = plt.subplots(figsize=(9.0, 5.4))
ax.axis("off")

def box(x, y, w, h, text, fc, ec, fs=11, tc=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018",
                 fc=fc, ec=ec, lw=1.8))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=tc or COL["ink"])

box(0.05, 0.54, 0.22, 0.20, "CLIP text\nencoder $T_\\theta$", "#eef1f4", "#aab4bf")
box(0.38, 0.54, 0.22, 0.20, r"embedding $C$", "#f7f9fb", "#aab4bf", fs=13)
box(0.71, 0.54, 0.22, 0.20, r"read-out $R=C M_\ell^{1/2}$", "#eaf5f2", COL["prop"], fs=12, tc=COL["prop"])
box(0.38, 0.16, 0.22, 0.20, "frozen UNet\ncross-attn", "#eef1f6", "#9aa7c4")
box(0.71, 0.16, 0.22, 0.20, r"metric $M_\ell^{1/2}$", "#f2eefb", COL["anchor"], fs=12, tc=COL["anchor"])

arrow(ax, (0.27, 0.64), (0.38, 0.64), COL["muted"], lw=2)
arrow(ax, (0.60, 0.64), (0.71, 0.64), COL["prop"], lw=2.4)
arrow(ax, (0.60, 0.26), (0.71, 0.26), COL["anchor"], lw=2.2)
arrow(ax, (0.82, 0.36), (0.82, 0.54), COL["anchor"], lw=2.0)
ax.annotate("gradient flows only to text encoder", xy=(0.16, 0.50), xytext=(0.16, 0.36),
            ha="center", fontsize=10.5, color=COL["prop"],
            arrowprops=dict(arrowstyle="-|>", color=COL["prop"], lw=1.8))
ax.text(0.49, 0.06, "UNet is frozen: it supplies the metric, not trainable weights",
        ha="center", color=COL["muted"], fontsize=11)
ax.set_title("LSSE geodesic erases what the frozen UNet actually reads", pad=16)
save(fig, "L1_readout_space")
plt.show()
""")

code(r"""# === Fig L2: geodesic target stays on manifold ============================
fig, ax = plt.subplots(figsize=(7.7, 6.7))
# Unit sphere / read-out manifold.
t = np.linspace(0, 2*np.pi, 260)
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.9)
ax.fill(np.cos(t), np.sin(t), color=COL["grid"], alpha=.45, zorder=0)

ang_z, ang_c = 1.22, 0.35
z0 = np.array([np.cos(ang_z), np.sin(ang_z)])
concept = np.array([np.cos(ang_c), np.sin(ang_c)])
eta = 0.58
geo_path = geodesic(z0, concept, -np.linspace(0, eta, 75))
z_geo = geo_path[-1]
z_linear = z0 - 0.95 * concept

# Concept direction and targets.
ax.plot(geo_path[:, 0], geo_path[:, 1], color=COL["prop"], lw=4)
arrow(ax, tuple(geo_path[-8]), tuple(z_geo), COL["prop"], lw=3.2)
arrow(ax, tuple(z0), tuple(z_linear), COL["forget"], lw=2.7, ls=(0, (4, 3)))
node(ax, z0, COL["axis"]); node(ax, concept, COL["forget"]); node(ax, z_geo, COL["prop"])
ax.scatter(*z_linear, s=90, color=COL["forget"], edgecolors="white", linewidths=1.4, zorder=8)

lbl(ax, (z0[0]-0.12, z0[1]+0.20), r"$R_0$", color=COL["axis"], fs=13)
lbl(ax, (concept[0]+0.18, concept[1]+0.14), r"$c_{read}$", color=COL["forget"], fc="#fdeef0", fs=12)
lbl(ax, (z_geo[0]-0.05, z_geo[1]+0.22), "geodesic target\n(on-manifold)", color=COL["prop"], fc="#eaf5f2", fs=11)
lbl(ax, (z_linear[0]-0.18, z_linear[1]-0.28), "linear push-away\n(off-manifold risk)", color=COL["forget"], fc="#fdeef0", fs=11)
ax.text(0, -1.34, r"$R_{target}=\mathrm{GeoAway}(R_0,c_{read},\eta)$", ha="center", fontsize=13,
        color=COL["prop"])

ax.set_title("LSSE geodesic rotates away from the concept without leaving the manifold")
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.35); ax.set_aspect("equal"); clean(ax)
save(fig, "L2_geodesic_target")
plt.show()
""")

code(r"""# === Fig L3: retain anchor in the same read-out metric ====================
fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.7), sharex=True, sharey=True,
                         constrained_layout=True)
rng = np.random.default_rng(7)
retain = rng.normal(loc=[-0.25, 0.15], scale=[0.22, 0.16], size=(30, 2))
concept = rng.normal(loc=[0.75, 0.55], scale=[0.18, 0.14], size=(18, 2))

def setup(ax, title):
    ax.scatter(retain[:, 0], retain[:, 1], s=36, color=COL["safe"], alpha=.65, label="retain frozen")
    ax.scatter(concept[:, 0], concept[:, 1], s=34, color=COL["forget"], alpha=.35, label="concept")
    ax.add_patch(plt.Circle((-0.25, 0.15), 0.44, ec=COL["safe"], fc="none", lw=1.8, ls=(0, (5, 3))))
    ax.set_title(title)
    ax.set_xlim(-1.1, 1.35); ax.set_ylim(-0.7, 1.15); ax.set_aspect("equal"); clean(ax)

setup(axes[0], "Anchor in the wrong space")
drift = retain + np.array([0.38, -0.28]) + rng.normal(scale=.04, size=retain.shape)
axes[0].scatter(drift[:, 0], drift[:, 1], s=34, color=COL["forget"], alpha=.65, label="retain current")
arrow(axes[0], (-0.25, 0.15), (0.13, -0.13), COL["forget"], lw=2.3)
lbl(axes[0], (0.34, -0.35), "retain drifts\nin UNet read-out", color=COL["forget"], fc="#fdeef0", fs=11)

setup(axes[1], "Read-out retain anchor")
cur = retain + rng.normal(scale=.045, size=retain.shape)
axes[1].scatter(cur[:, 0], cur[:, 1], s=34, color=COL["prop"], alpha=.75, label="retain current")
axes[1].scatter(-0.25, 0.15, facecolors="none", edgecolors=COL["anchor"], s=980, lw=2.4)
lbl(axes[1], (-0.13, -0.36), r"$\mathrm{mean}_\ell\,\|R_\ell(z)-R_\ell(z_0)\|^2$",
    color=COL["anchor"], fc="#f2eefb", fs=11)

for ax in axes:
    ax.legend(loc="upper left")
fig.suptitle("Retain must be anchored in the same read-out space as erasure",
             fontsize=14, fontweight="bold", color=COL["ink"])
save(fig, "L3_readout_retain_anchor")
plt.show()
""")

code(r"""# === Fig L4: PLU - progressive layer unlocking =============================
# Actual trainer schedule (lsse_trainer.py): start with 1 trainable layer, then
# 3 at int(E/3)+1 and 6 at int(2E/3)+1 — EARLY (low-index, CAP-causal) layers first.
fig, ax = plt.subplots(figsize=(9, 4.9))
n_layers, n_epochs = 12, 60
unlock_mid, unlock_full = n_epochs // 3 + 1, 2 * n_epochs // 3 + 1

def k_of(e):
    return 6 if e >= unlock_full else (3 if e >= unlock_mid else 1)

sched = np.zeros((n_layers, n_epochs))
for e in range(1, n_epochs + 1):
    sched[:k_of(e), e - 1] = 1
ax.imshow(sched, aspect="auto", cmap="BuGn", origin="lower",
          extent=[1, n_epochs + 1, 0, n_layers], vmin=0, vmax=1.35)
ks = [k_of(e) for e in range(1, n_epochs + 1)]
ax.step(range(1, n_epochs + 1), ks, color=COL["forget"], lw=2.6, where="post")
ax.set_xlabel("epoch")
ax.set_ylabel("trainable text-encoder layers\n(early / CAP-causal first)")
ax.set_title("PLU: 1 → 3 → 6 layers, unlocked at 1/3 and 2/3 of training")
lbl(ax, (14, 9.2), "frozen (upper layers stay frozen)", color=COL["muted"], fs=11)
lbl(ax, (50, 2.2), "trainable", color=COL["prop"], fc="#eaf5f2", fs=11)
ax.text(30, -1.75, "12-layer CLIP TE — LSSE geodesic trains at most the first 6 layers (L0–L5)",
        ha="center", color=COL["muted"], fontsize=10.5)
save(fig, "L4_progressive_layer_unlocking")
plt.show()
""")

md(r"""## Summary — three verbs, one goal""")

code(r"""# === Fig SUM: ODACE redirect - Sph+OT move - LSSE geodesic ===============
fig, axes = plt.subplots(1, 3, figsize=(14, 3.45), constrained_layout=True)

# ODACE
ax = axes[0]
e0 = np.array([0, 0]); ep = np.array([1.05, 0.78]); eb = np.array([-0.62, 0.48]); et = eb - 0.38*(ep-eb)
arrow(ax, tuple(e0), tuple(ep), COL["forget"])
arrow(ax, tuple(e0), tuple(eb), COL["safe"])
arrow(ax, tuple(eb), tuple(et), COL["prop"])
node(ax, e0, COL["axis"], s=55); node(ax, eb, COL["safe"], s=70); node(ax, et, COL["prop"], s=70)
ax.set_title("ODACE - redirect output", color=COL["prop"])
lbl(ax, (-0.02, -0.35), r"$\epsilon_b-\lambda(\epsilon_p-\epsilon_b)$", fs=11)
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-0.55, 1.12)

# Sph+OT
ax = axes[1]
t = np.linspace(0, np.pi, 200)
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.8)
a0, a1 = 0.45, 1.4
p_x = np.array([np.cos(a1), np.sin(a1)]); p_c = np.array([np.cos(a0), np.sin(a0)])
arc = np.array([exp_map(p_x, -s*0.7*log_map(p_x, p_c)) for s in np.linspace(0, 1, 30)])
g, = ax.plot(arc[:, 0], arc[:, 1], color=COL["prop"], lw=3.2); glow(g)
node(ax, p_x, COL["axis"]); node(ax, p_c, COL["forget"])
ax.set_title("Sph+OT - move on sphere", color=COL["prop"])
lbl(ax, (0, -0.35), r"$\mathrm{Exp}_{u_x}(-\mu_p\,\mathrm{Log}_{u_x}(u_c))$", fs=10.5)
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.55, 1.15)

# LSSE geodesic
ax = axes[2]
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.8)
rz = np.array([np.cos(1.25), np.sin(1.25)]); rc = np.array([np.cos(0.35), np.sin(0.35)])
geo = geodesic(rz, rc, -np.linspace(0, .55, 40))
ax.plot(geo[:, 0], geo[:, 1], color=COL["prop"], lw=3.2)
node(ax, rz, COL["axis"]); node(ax, rc, COL["forget"]); node(ax, geo[-1], COL["prop"])
ax.set_title("LSSE - geodesic read-out", color=COL["prop"])
lbl(ax, (0, -0.35), r"$R=C M_\ell^{1/2};\; R\to\mathrm{GeoAway}(R,c)$", fs=10)
ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.55, 1.15)

for ax in axes:
    ax.set_aspect("equal")
    clean(ax)
fig.suptitle("Three spaces - three verbs - one goal (erase concept, keep coherence)",
             fontsize=15, fontweight="bold", color=COL["ink"])
save(fig, "SUM_comparison")
plt.show()
""")

md(r"""## Bonus — animated process figures (GIF)

High-impact for talks: these show the *process*, not just the end state. Each
cell saves an animated `.gif` into `figures/` (bundled in the final zip) and
plays inline in Colab. Drop the GIF straight into Google Slides / Keynote.""")

code(r"""# === Animation helpers ======================================================
from matplotlib.animation import FuncAnimation, PillowWriter
try:
    from IPython.display import Image as IPyImage      # Colab: plays GIF inline
except ModuleNotFoundError:                            # headless fallback
    def IPyImage(filename=None, **_):
        return filename

def save_gif(anim, name, fps=18):
    path = f"figures/{name}.gif"
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(anim._fig)
    print("saved", path)
    return path

print("animation helpers ready")""")

code(r"""# === Anim A1: ODACE - benign redirect avoids collapse =====================
fig, ax = plt.subplots(figsize=(7.4, 5.9))
forget = np.array([1.4, -1.15]); benign = np.array([-1.35, -1.05]); collapse = np.array([0.1, -1.85]); start = np.array([0.0, 1.85])

def _bez(p0, p1, p2, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2

base = _bez(start, np.array([0.9, 0.3]), forget)
push = _bez(start, np.array([0.0, -0.1]), collapse)
redir = _bez(start, np.array([-0.9, 0.3]), benign)
for c, col, lab in [(forget, COL["forget"], "concept mode"), (benign, COL["safe"], "benign mode"), (collapse, COL["base"], "collapse")]:
    ax.add_patch(plt.Circle(tuple(c), 0.55, color=col, alpha=.14, zorder=0)); node(ax, c, col)
    ax.text(c[0], c[1]-0.88, lab, ha="center", color=col, fontsize=11)
node(ax, start, COL["axis"]); lbl(ax, (start[0]+0.32, start[1]), r"$z_T$", fs=12)
lB, = ax.plot([], [], color=COL["base"], lw=3.0, label="raw"); glow(lB)
lP, = ax.plot([], [], color=COL["forget"], lw=2.7, label="old push-away"); glow(lP)
lR, = ax.plot([], [], color=COL["prop"], lw=3.4, label="benign redirect"); glow(lR)
ax.legend(loc="upper right")
ax.set_title("ODACE redirects toward a coherent benign basin")
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.45, 2.35); ax.set_aspect("equal"); clean(ax)

N = len(base); F = 60
def _upd(f):
    k = max(2, int(N*(f+1)/F))
    lB.set_data(base[:k, 0], base[:k, 1])
    lP.set_data(push[:k, 0], push[:k, 1])
    lR.set_data(redir[:k, 0], redir[:k, 1])
    return lB, lP, lR

anim = FuncAnimation(fig, _upd, frames=F, interval=50, blit=False)
p = save_gif(anim, "A1_odace_benign_redirect", fps=20)
IPyImage(filename=p)
""")

code(r"""# === Anim A2: Sph+OT — geodesic slides on the arc, Euclidean shoots off =====
fig, ax = plt.subplots(figsize=(7.2, 5.9))
t = np.linspace(np.pi*0.03, np.pi*0.97, 220)
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=2)
ax.fill_between(np.cos(t), np.sin(t), 0, color=COL["grid"], alpha=.45, zorder=0)
ux = np.array([np.cos(1.15), np.sin(1.15)]); uc = np.array([np.cos(0.35), np.sin(0.35)]); mu = 0.7
node(ax, ux, COL["axis"]); node(ax, uc, COL["forget"])
lbl(ax, (ux[0]-0.02, ux[1]+0.15), r"$u_x$", color=COL["axis"], fs=12)
lbl(ax, (uc[0]+0.12, uc[1]+0.08), r"$u_c$", color=COL["forget"], fs=12)
v = log_map(ux, uc)
trail, = ax.plot([], [], color=COL["prop"], lw=3.6); glow(trail)
te, = ax.plot([], [], color=COL["base"], lw=2.4, ls=(0, (4, 3)))
dg = ax.scatter([ux[0]], [ux[1]], s=95, color=COL["prop"], edgecolors="white", zorder=9)
de = ax.scatter([ux[0]], [ux[1]], s=95, color=COL["base"], edgecolors="white", zorder=9)
ax.set_title("Geodesic stays on the arc;  Euclidean leaves it")
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.2, 1.28); ax.set_aspect("equal"); clean(ax)

NF = 50
def _upd(f):
    s = f/(NF-1)
    g = np.array([exp_map(ux, -a*s*mu*v) for a in np.linspace(0, 1, 30)])
    trail.set_data(g[:, 0], g[:, 1])
    gp = exp_map(ux, -s*mu*v); ep = ux - s*mu*uc
    te.set_data([ux[0], ep[0]], [ux[1], ep[1]])
    dg.set_offsets([gp]); de.set_offsets([ep])
    return trail, te, dg, de

anim = FuncAnimation(fig, _upd, frames=NF, interval=60, blit=False)
p = save_gif(anim, "A2_sph_geodesic", fps=18)
IPyImage(filename=p)""")

code(r"""# === Anim A3: LSSE - geodesic read-out rotation ===========================
fig, ax = plt.subplots(figsize=(6.9, 6.4))
t = np.linspace(0, 2*np.pi, 260)
ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.9)
ax.fill(np.cos(t), np.sin(t), color=COL["grid"], alpha=.45, zorder=0)
concept = np.array([np.cos(0.35), np.sin(0.35)])
z0 = np.array([np.cos(1.25), np.sin(1.25)])
path_geo = geodesic(z0, concept, -np.linspace(0, .72, 80))
node(ax, concept, COL["forget"]); lbl(ax, (concept[0]+0.20, concept[1]+0.13), r"$c_{read}$", color=COL["forget"], fc="#fdeef0", fs=12)
node(ax, z0, COL["axis"]); lbl(ax, (z0[0]-0.12, z0[1]+0.20), r"$R_0$", color=COL["axis"], fs=12)
ln, = ax.plot([], [], color=COL["prop"], lw=3.5)
pt = ax.scatter([z0[0]], [z0[1]], s=85, color=COL["prop"], edgecolors="white", zorder=9)
txt = ax.text(0, -1.28, "", ha="center", fontsize=13, color=COL["prop"])
ax.set_title("LSSE geodesic rotates read-out embeddings away from the concept")
ax.set_xlim(-1.45, 1.45); ax.set_ylim(-1.45, 1.35); ax.set_aspect("equal"); clean(ax)

NF = len(path_geo)
def _upd(f):
    cur = path_geo[:f+1]
    ln.set_data(cur[:, 0], cur[:, 1])
    pt.set_offsets([path_geo[f]])
    txt.set_text(r"on-manifold geodesic step  $\eta=%.2f$" % (0.72 * f/(NF-1)))
    return ln, pt, txt

anim = FuncAnimation(fig, _upd, frames=NF, interval=55, blit=False)
p = save_gif(anim, "A3_lsse_geodesic", fps=18)
IPyImage(filename=p)
""")

code(r"""# === Bundle presentation-ready figures for download (Colab) ===============
from pathlib import Path
import zipfile

fig_dir = Path("figures")
zip_path = Path("method_figures.zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fp in sorted(fig_dir.iterdir()):
        if not fp.is_file() or fp.name.startswith("_mid_"):
            continue
        zf.write(fp, arcname=fp.name)
print("zipped -> method_figures.zip")
try:
    from google.colab import files  # type: ignore
    files.download(str(zip_path))
except Exception as e:
    print("(not in Colab) figures are in ./figures/ -", e)
""")

def to_cell(kind, src):
    base = {"metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "md":
        base["cell_type"] = "markdown"
    else:
        base.update(cell_type="code", execution_count=None, outputs=[])
    return base


nb = {
    "cells": [to_cell(k, s) for k, s in CELLS],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", OUT, "-", len(CELLS), "cells")
