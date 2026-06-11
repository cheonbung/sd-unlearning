"""Generate `method_visualizations.ipynb` (Colab-ready) from stdlib json only.

High-quality, slide-ready explanatory figures for the three proposed methods
(ODACE / Sph+OT / LSSE+PLU+W2), each contrasted with the CONVENTIONAL
(Euclidean / text-proxy) approach. Pure matplotlib with a consistent design
system (serif + LaTeX mathtext, 220 dpi, curated palette, rounded callouts,
soft depth) so every PNG drops cleanly into PPT.

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


md(r"""# Concept-Unlearning Methods — Explanatory Figures (HD)

Slide-ready intuition for the three proposed methods, each contrasted with the
**conventional** baseline it improves on.

| Method | Space | Verb | Core operation |
|---|---|---|---|
| **ODACE** | noise $\epsilon$ | *bend the output* | $\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$ |
| **Sph+OT** | sphere $\mathbb{S}^{d-1}$ | *move on the sphere* | $\mathrm{Exp}_{u_x}(-\mu_p\,\mathrm{Log}_{u_x}(u_c))$ |
| **LSSE+PLU+W2** | embedding $\mathbb{R}^{d}$ | *project the axis out* | $(z\cdot c_{dir})^2\!\to\!0$ + $\perp$ anchor |

**Run all** (`Runtime ▸ Run all`). Each code cell renders one HD figure (220 dpi)
into `figures/`; the last cell zips them. Labels are English + LaTeX (no extra
fonts needed); a Korean-font cell is provided if you want Korean labels.""")


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
    anchor="#7b62c4",    # W2 anchor (violet)
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

# Panel 1: noise space (swirling epsilon field)
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-1, 1, 11), np.linspace(-1, 1, 11))
ax.quiver(gx, gy, -gy, gx, color=COL["prop"], alpha=.5, width=.006, scale=24)
ax.set_title("ODACE", color=COL["prop"])
lbl(ax, (0, 1.04), r"$\epsilon\in\mathbb{R}^{C\times H\times W}$", fc="#eef6f4",
    ec=COL["prop"], fs=12)
ax.text(0, -1.34, "noise space — bend the output", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25)

# Panel 2: sphere with a geodesic
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
ax.text(0, -1.4, "manifold — move on the sphere", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.45, 1.3)

# Panel 3: embedding null-space
ax = axes[2]
ax.axhline(0, color=COL["grid"], lw=1.4); ax.axvline(0, color=COL["grid"], lw=1.4)
cd = np.array([1, 1]) / np.sqrt(2)
ax.plot([-1.25*cd[1], 1.25*cd[1]], [1.25*cd[0], -1.25*cd[0]], color=COL["prop"], lw=3.4)
arrow(ax, (0, 0), tuple(cd*1.15), COL["forget"])
lbl(ax, tuple(cd*1.34), r"$c_{dir}$", color=COL["forget"], fc="#fdeef0", fs=12)
ax.set_title("LSSE", color=COL["prop"])
lbl(ax, (0, 1.16), r"$z\in\mathbb{R}^{d}$", fc="#eef6f4", ec=COL["prop"], fs=12)
ax.text(0, -1.4, "embedding — project the axis out", ha="center",
        fontsize=11, color=COL["muted"])
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.45, 1.3)

for ax in axes:
    ax.set_aspect("equal"); clean(ax)
fig.suptitle("Unlearning re-framed: which space do we optimize in?",
             fontsize=16, fontweight="bold", color=COL["ink"])
save(fig, "00_overview_spaces")
plt.show()""")


md(r"""## ODACE — output-grounded erasure in noise space
$$\epsilon_{target}=\epsilon_0-\eta\,(\epsilon_p-\epsilon_0),\qquad
L=\mathbb{E}\,\big\|\epsilon_\theta(z_t,t,c_{forget})-\epsilon_{target}\big\|_2^2.$$""")


md(r"""### 📑 구성 요소 & 기호 (Notation)

ODACE는 CLIP text encoder를 **frozen**으로 두고, UNet의 **cross-attention projection만** 학습합니다 — text 임베딩(proxy)이 아니라 **출력 noise $\epsilon$** 를 직접 감독합니다.

| 기호 | 의미 |
|---|---|
| $z_t$ | diffusion timestep $t$의 latent(잡음 섞인 중간 이미지) |
| $\epsilon_\theta(z_t,t,c)$ | UNet이 latent·timestep·conditioning $c$에서 예측한 noise |
| $\epsilon_0$ | unconditional($c_0$) 예측 — 개념 없는 기준점 |
| $\epsilon_p$ | concept(forget) prompt 예측 |
| $\epsilon_p-\epsilon_0$ | concept가 denoising에 **추가한 방향** |
| $\eta$ | erase 강도(반대로 미는 세기; v3는 $\eta{=}3.0$) |
| $\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$ | **negative-guidance 타깃**(개념 방향의 반대) |
| $c_{forget},\,c_{retain},\,c_0$ | forget / retain / unconditional conditioning |

| 손실 | 식 | 역할 |
|---|---|---|
| $L_{forget}$ | $\lVert\epsilon_\theta(z_t,t,c_{forget})-\epsilon_{target}\rVert^2$ | 개념 예측을 반대 타깃으로 |
| $L_{retain}$ | $\lVert\epsilon_\theta(z_t,t,c_{retain})-\epsilon_{frozen}\rVert^2$ | 일반 prompt는 원본 UNet과 동일하게(locality) |
| $L_{total}$ | $\alpha\,L_{forget}+\beta\,L_{retain}$ | 두 항의 균형 |

**변형(variant)**

| 변형 | 학습 대상 | 결과 |
|---|---|---|
| ODACE **v2** | cross-attn **K/V만**(to_k, to_v) | 약한 편집 — ASR≈raw (**negative result**) |
| ODACE **v3** | cross-attn **full**(to_q/k/v/out), $\eta{=}3$, 1500 step | **ASR 4.0** — 최저 |
| ODACE **v1.5** | v3 recipe를 SD v1.5 base에 적용 | 전이 확인(우연한 약점이 아님) |""")


md(r"""### 💡 직관 보강 — ESD와 무엇이 다른가 · 두 표현 풀이

> **⚠️ 먼저 — 'conventional'은 그림마다 다릅니다.** 위 그림 **O3**의 'Conventional'은 **text-proxy 계열**(FCF·LSSE·DACE·Sph+OT)로, 이들은 CLIP **text encoder**를 고치고 **UNet은 frozen**입니다. 아래 Q1의 **ESD**는 이와 **다른** baseline으로, **UNet**(cross-attn 제외)을 고칩니다. 즉 ODACE는 *두 종류의 conventional*과 대비됩니다 — text-proxy(O3) **및** ESD(Q1). ODACE는 둘 다와 달리 "UNet을 고치되 **cross-attn만**".

**Q1. ESD와 학습 목표가 같아 보이는데, 구체적으로 뭐가 다른가?**
맞습니다 — **타깃 식은 동일**합니다: 둘 다 $\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$ 에 MSE + retain. 차이는 **'어디를' 편집하느냐**입니다.

| | ESD-u | ODACE v3 |
|---|---|---|
| 편집 위치 | cross-attn을 **제외한**(`noxattn`) 광범위 UNet 파라미터 | cross-attn **만**(to_q/k/v/out) |
| 침습도 | 넓음·무거움(많은 UNet weight) | 좁음·국소(작은 서브모듈) |
| 개념과의 직접성 | 시각 경로를 바꿈 — text 조건과 **간접** | text→이미지 **조건 주입 지점**을 직접 |
| recipe | canonical | $\eta{=}3$ 강한 적대 push + retain locality |

→ 두 방법은 위치가 거의 **상보적**입니다(ESD = cross-attn '빼고', ODACE = cross-attn '만'). ODACE의 주장은 "UNet을 많이 건드리자"가 아니라 **"개념이 이미지로 들어가는 단 하나의 관문(cross-attn)을 출력 기준으로 누르자"** 입니다. 그래서 더 적은 파라미터로 더 낮은 ASR + 유틸리티 보존을 얻습니다.

**Q2. 'full cross-attention을 국소 편집한다'가 무슨 뜻?**
- **cross-attention** = text 임베딩이 이미지 feature에 주입되는 UNet 내부 attention 블록. 투영행렬 4개로 구성: query `to_q`, key `to_k`, value `to_v`, output `to_out`.
- **full** = 그 4개 투영을 **모두** 학습(v3). K/V 2개만 연 v2는 너무 약해 실패 → 관문을 '전부' 열어야 효과가 남.
- **국소(local)** = 그런데 **cross-attn 블록 안으로만** 한정 — conv·self-attn·time-embedding 등 UNet의 **나머지는 전부 frozen**. UNet 전체 파라미터의 극히 일부만 움직입니다.
- 즉 "**cross-attn은 통째로(full), 그러나 cross-attn에만 국한(local)**". 효과적(조건 관문을 장악)이면서 안전(나머지 보존)한 이유가 여기 있습니다.

**Q3. '디노이징 궤적을 출발점에서부터 휘게 만들어 개념 모드를 벗어난다'를 풀어주면?**
- **디노이징 궤적** = 순수 노이즈 $z_T$에서 시작해 매 step UNet이 예측한 $\epsilon$만큼 잡음을 걷어내며 $z_0$(이미지)로 가는 **경로**.
- **개념 모드(concept mode)** = 출력 분포에서 그 개념(nudity) 이미지들이 모인 **영역(basin)**. concept prompt를 주면 궤적이 보통 이 basin으로 흘러듭니다.
- **반대 타깃의 효과** = $\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$ 은 매 step의 예측을 개념 쪽이 아니라 **반대쪽**으로 틀어줍니다($\eta{=}3$이면 강하게 overshoot).
- **'출발점에서부터'의 두 의미**
  1. *시간적*: $\epsilon$은 **가장 이른(고노이즈) step부터** 적용됩니다. 그림의 큰 구도·모드가 결정되는 초반에 방향을 틀면, 그 편차가 역과정 전체에 누적되어 궤적이 **다른 basin**으로 빠집니다. (이미 다 그려진 뒤 늦게 밀어선 basin을 못 벗어남.)
  2. *인과적*: ODACE는 개념이 **들어오는 관문(cross-attn)** 을 고치므로, 영향의 **발원지**에서 바로 휘게 합니다 — 하류에서 사후 보정하는 게 아니라.
- 결과: 궤적이 처음부터 개념 basin을 벗어나 **안전한 모드**로 수렴 → 이미지 수준에서 개념이 사라집니다. (그림 **O2** 정적 + **A1** 애니메이션이 이 휘어짐을 보여줍니다.)""")


code(r"""# === Fig O1: negative-guidance vector decomposition =========================
fig, ax = plt.subplots(figsize=(7.6, 6.2))
e0 = np.array([0.0, 0.0]); ep = np.array([1.7, 1.1]); eta = 0.5
d = ep - e0; etarget = e0 - eta*d

arrow(ax, tuple(e0), tuple(d+e0), COL["base"], lw=1.8, ls=(0, (4, 3)))
arrow(ax, tuple(e0), tuple(ep), COL["forget"], lw=3)
arrow(ax, tuple(e0), tuple(etarget), COL["prop"], lw=3)
node(ax, e0, COL["axis"])

lbl(ax, (-0.42, -0.18), r"$\epsilon_0$", color=COL["axis"], fs=13)
lbl(ax, (ep[0]+0.05, ep[1]+0.12), r"$\epsilon_p=\mathrm{UNet}(z_t,t,c_{forget})$",
    color=COL["forget"], fc="#fdeef0", fs=12)
lbl(ax, (d[0]+0.15, d[1]-0.30), r"$\epsilon_p-\epsilon_0$  (forget dir.)",
    color=COL["muted"], fs=11)
lbl(ax, (etarget[0]-0.35, etarget[1]-0.30),
    r"$\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$",
    color=COL["prop"], fc="#eef6f4", fs=12)

ax.set_title("ODACE target = reverse of the forget direction")
ax.set_xlim(-1.7, 2.3); ax.set_ylim(-1.2, 1.7); ax.set_aspect("equal"); clean(ax)
save(fig, "O1_negative_guidance_vectors")
plt.show()""")


code(r"""# === Fig O2: denoising trajectory bent away from the forget mode =============
fig, ax = plt.subplots(figsize=(7.8, 6.2))
forget = np.array([1.4, -1.2]); safe = np.array([-1.4, -1.2]); start = np.array([0.0, 1.85])

def bezier(p0, p1, p2, n=90):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2

base = bezier(start, np.array([0.95, 0.3]), forget)
odace = bezier(start, np.array([-0.95, 0.3]), safe)

for c, col, lab in [(forget, COL["forget"], "forget mode\n(e.g. nudity)"),
                    (safe, COL["safe"], "safe mode")]:
    ax.add_patch(plt.Circle(tuple(c), 0.6, color=col, alpha=.14, zorder=0))
    node(ax, c, col); ax.text(c[0], c[1]-0.95, lab, ha="center", color=col, fontsize=11)

lb, = ax.plot(base[:, 0], base[:, 1], color=COL["base"], lw=3.4, label="baseline"); glow(lb)
lo, = ax.plot(odace[:, 0], odace[:, 1], color=COL["prop"], lw=3.4, label="ODACE"); glow(lo)
arrow(ax, tuple(base[-9]), tuple(base[-1]), COL["base"], lw=3.4)
arrow(ax, tuple(odace[-9]), tuple(odace[-1]), COL["prop"], lw=3.4)
node(ax, start, COL["axis"]); lbl(ax, (start[0]+0.28, start[1]+0.04), r"$z_T$", fs=12)

ax.set_title("Reverse-$\\eta$ target bends the trajectory at its source")
ax.legend(loc="upper right")
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.3, 2.35); ax.set_aspect("equal"); clean(ax)
save(fig, "O2_trajectory_bending")
plt.show()""")


code(r"""# === Fig O3: conventional text-proxy  vs  ODACE output supervision ==========
def card(ax, x, y, w, h, text, fc, ec, fs=11, tc=None):
    ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.018",
                 fc=fc, ec=ec, lw=1.8, mutation_aspect=1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fs,
            color=tc or COL["ink"])

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

ax = axes[0]; ax.set_title("Conventional — edit the TEXT proxy", color=COL["base"])
card(ax, 0.04, 0.55, 0.24, 0.2, "prompt $c$", "#eef1f4", "#aab4bf")
card(ax, 0.38, 0.5, 0.26, 0.3, "edit $T_\\theta$\n(text space)", "#fdeef0", COL["forget"],
     tc=COL["forget"])
card(ax, 0.74, 0.5, 0.22, 0.3, "UNet\n(frozen)", "#eef1f6", "#9aa7c4")
arrow(ax, (0.28, 0.65), (0.38, 0.65), COL["muted"], lw=2)
arrow(ax, (0.64, 0.65), (0.74, 0.65), COL["muted"], lw=2)
ax.annotate("", xy=(0.85, 0.46), xytext=(0.85, 0.34),
            arrowprops=dict(arrowstyle="-|>", color="#c3c9cf", lw=1.6))
lbl(ax, (0.5, 0.24), "image effect is INDIRECT  →  leakage", color=COL["forget"],
    fc="#fdeef0", fs=11)

ax = axes[1]; ax.set_title("ODACE — supervise the OUTPUT $\\epsilon$", color=COL["prop"])
card(ax, 0.04, 0.55, 0.24, 0.2, "prompt $c_{forget}$", "#eef1f4", "#aab4bf", fs=10)
card(ax, 0.38, 0.5, 0.26, 0.3, "UNet cross-attn\n$W_{Q,K,V,O}$", "#eaf5f2", COL["prop"],
     tc=COL["prop"])
card(ax, 0.74, 0.5, 0.22, 0.3, r"$\epsilon_\theta$", "#eaf5f2", COL["prop"], fs=16,
     tc=COL["prop"])
arrow(ax, (0.28, 0.65), (0.38, 0.65), COL["muted"], lw=2)
arrow(ax, (0.64, 0.65), (0.74, 0.65), COL["muted"], lw=2)
ax.annotate("", xy=(0.85, 0.46), xytext=(0.85, 0.34),
            arrowprops=dict(arrowstyle="-|>", color=COL["prop"], lw=2))
lbl(ax, (0.85, 0.26), r"$\|\epsilon_\theta-\epsilon_{target}\|^2$", color=COL["prop"],
    fc="#eaf5f2", fs=12)
ax.text(0.5, 0.1, "DIRECT supervision on the generated noise", ha="center",
        color=COL["prop"], fontsize=11)

for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0.03, 0.92); ax.axis("off")
save(fig, "O3_proxy_vs_output")
plt.show()""")


code(r"""# === Fig O4: locality — only cross-attention is trained =====================
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
ax.text(0.46, 0.8, "UNet — frozen except cross-attention", ha="center", fontsize=14,
        fontweight="bold", color=COL["ink"])
ax.text(0.46, 0.1, "tiny trainable footprint  →  erase the concept, keep image quality",
        ha="center", color=COL["muted"], fontsize=10.5)
ax.set_xlim(0, 0.98); ax.set_ylim(0.03, 0.9); ax.axis("off")
save(fig, "O4_crossattn_locality")
plt.show()""")


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
| $z^{\star}=\arg\max_{z_i}\mathcal{W}$ | worst-case noise | $\mathcal{W}$를 최대화하는(개념을 가장 드러내는) 노이즈 |
| $\omega$ | projection dir. | 1D 사영(슬라이싱) 방향 |
| $F^{-1}$ | quantile func. | 역누적분포(1D $W_2$ 닫힌 형에 사용) |

**대조 — FCF (기존 연구)**

| 기호 / 수식 | 의미 |
|---|---|
| $\mathrm{proj}_c(x)=\langle x,u_c\rangle\,u_c$ | 개념 축으로의 사영 성분 |
| $x_{clean}=\dfrac{x-\eta\,\mathrm{proj}_c(x)}{1-\eta}$ | FCF 정제식 (Euclidean, **현**을 따라감) |
| $\eta$ | FCF 삭제 비율(여기선 직선 보간 계수) |

> $\mu_p$(Sph: 측지 *각도* 비율)와 $\eta$(FCF: 직선 *보간* 계수)는 이름은 비슷하지만 **역할이 다릅니다** — 같은 '삭제 강도'라도 한쪽은 각도를, 한쪽은 직선 거리를 조절합니다.""")


md(r"""### 💡 직관 보강 — 자주 나오는 의문

**Q1. 왜 하필 '단위구(unit sphere)'인가?**
CLIP은 코사인 유사도 $\cos(x,c)=\langle x,c\rangle/(\lVert x\rVert\,\lVert c\rVert)$ 로 학습됩니다 — **크기는 무시하고 방향만** 비교하죠. 그래서 의미는 벡터의 길이가 아니라 *방향*에 담기고, 모든 임베딩은 사실상 **반지름이 고정된 구면 위의 점**으로 취급됩니다. "개념을 지운다 = 구면 위에서 그 방향으로부터 멀어진다." 반지름은 어차피 임의이므로 **1로 고정**(= 단위구)하는 것이 표준 관례입니다.

**Q2. '현(chord)을 따라간다'는 게 — 무엇이, 언제?**
개념 성분을 빼는 편집 스텝에서 **텍스트 임베딩 점 그 자체**가 움직입니다. FCF의 $x-\eta\,\mathrm{proj}_c(x)$ 는 삭제 강도를 키울수록 점을 **직선으로** 끌고 가는데, 이 직선이 구 내부를 가로지르는 **지름길(현/할선)** 입니다. → 끝점이 구 **안쪽**으로 떨어져 **노름이 줄어듦**(그림 S2). 측지선은 표면을 따라 도는 **호(arc)** 라서 끝점이 구 위에 그대로 남습니다(그림 S1).

| | 경로 | 끝점 노름 | 위치 |
|---|---|---|---|
| FCF (직선 빼기) | 현 / 할선 | 줄어듦 (<1) | 구 **안쪽** ✗ |
| Sph+OT (측지선) | 호(arc) | 보존 (=1) | 구 **표면** ✓ |

*2D 예시* ($u_x$ 각 $60°$, 개념축 $x$, 강도 $0.5$): FCF $\to(0.25,\,0.87)$ 노름 **0.90**(안쪽); 측지선 $\to(0.26,\,0.97)$ 노름 **1.00**(표면). 둘 다 개념 성분은 줄이지만 끝나는 위치가 다릅니다.

**Q3. 그럼 FCF의 $/(1-\eta)$ 는 무엇인가?**
구 안으로 빠진 점을 **반지름 방향으로 다시 늘려** 구 쪽으로 밀어주는 보정입니다. 하지만 반지름 스케일링 $\neq$ 측지선 회전이라, 보정 후에도 측지선이 닿는 **정확한 표면 점과는 다른 위치**에 떨어집니다 — '현으로 간 뒤 억지로 표면에 붙이기' vs '처음부터 표면을 따라가기'의 차이입니다.

**Q4. OT는 왜 필요한가?**
구면 편집(Sph)만으로는 *평균적으로* 개념을 지우지만, **특정 노이즈**에서 개념이 새어나올 수 있습니다. OT는 후보 노이즈 중 개념을 가장 잘 드러내는 **worst-case $z^{\star}$** 를 골라 그에 강건하게 학습해 ASR을 한 번 더 끌어내립니다(그림 S3).""")


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
ax.legend(loc="upper left"); ax.set_aspect("equal"); clean(ax)
save(fig, "S3_optimal_transport_noise")
plt.show()""")


md(r"""## LSSE+PLU+W2 — null-space projection + orthogonal anchoring + PLU
$$L_{CNP}=\mathbb{E}\,(T_\theta(p)\cdot c_{dir})^2\to0,\qquad
L_{W2}=L_{CNP}+\lambda\,\mathbb{E}\,\|z^{\perp}_{cur}-z^{\perp}_{frozen}\|_2^2.$$""")


md(r"""### 📑 구성 요소 & 기호 (Notation)

LSSE는 FCF와 달리 **2-stage가 아니라 단일 학습 루프**에서 forget·implicit·retain을 함께 처리하고, FCF의 **임의 noise prompt를 쓰지 않습니다**. 세 축(CNP·CSR·CLM) + 두 변형(PLU·W2)으로 구성됩니다.

| 약어 | 풀네임 | 하는 일 | 왜 사용했나 |
|---|---|---|---|
| **CNP** | Concept Null-Space Projection | SVD로 개념 축 $c_{dir}$를 뽑아 $z\!\cdot\!c_{dir}$ 성분을 0으로 | FCF의 임의 noise 목적지를 **기하학적 null-space 목표**로 대체 |
| **CSR** | Contrastive Semantic Retention | retain 임베딩을 frozen과 InfoNCE로 정렬 | MSE보다 retain **공간 구조**를 더 잘 보존 |
| **CLM** | CAP-Guided Layer Masking | CAP 상위 layer만 학습, 나머지 freeze | 개입을 **개념 관련 layer로 국소화** → 유틸리티 보존 |
| **PLU** | Progressive Layer Unlocking | 1→3→6 layer로 점진 개방 | 초반 downstream 오염 완화 — LSSE 최대 개선폭 |
| **W2** | margin CNP (orthogonal anchoring) | 직교 성분 $z^{\perp}$를 frozen에 약하게 고정 | **concept rerouting**(옆문으로 새는 것) 억제 |

**기호**

| 기호 | 의미 |
|---|---|
| $T_\theta(p),\ T_0(p)$ | 학습 중 / frozen 원본 text encoder가 prompt $p$를 인코딩한 임베딩 |
| $c_{dir}$ | explicit 임베딩의 SVD 1주성분 = '개념 축' 단위벡터 |
| $z\!\cdot\!c_{dir}$ | 임베딩이 개념 축에 걸친 정도(스칼라) — CNP가 0으로 |
| $z^{\perp}_{cur},\ z^{\perp}_{frozen}$ | 개념 축과 **직교**한 성분(현재 / 원본) |
| $\lambda$ | W2 직교 앵커 강도 (크면 못 지움, 작으면 rerouting 못 막음) |
| $\mathrm{CAP}$ | layer별 **개념 인과 중요도** 점수(heatmap) — 학습 layer 선택·순서에 사용 |

> **CAP**는 '어느 CLIP layer가 개념 인코딩에 인과적으로 중요한가'를 layer별 점수로 매긴 프로젝트의 사전 분석(heatmap)입니다. **CLM**은 이 점수로 *학습할 layer*를, **PLU**는 여는 *순서*를 정합니다.""")


md(r"""### 💡 직관 보강 — 유틸리티 & FCF 2-stage 대비 우위

**Q1. 여기서 '유틸리티(utility)'란?**
지우려는 개념(nudity)이 **아닌** 일반 콘텐츠에 대한 모델의 **쓸모** — retain 프롬프트로 정상적이고 품질 좋은 이미지를 그대로 생성하는 능력입니다. 정량적으로는 **COCO FID**(이미지 품질·충실도), **COCO CLIP**(프롬프트-이미지 정렬), **IQ**로 측정합니다.

**Q2. '유틸리티 보존'은 무엇을 의미하나?**
개념은 지우되(↓ASR) 일반 생성 성능은 **떨어뜨리지 않는 것**. 공격적으로 지우면 보통 일반 품질이 함께 망가지는 trade-off가 생기는데, LSSE는 세 장치로 이를 막습니다 — **CSR**(retain 구조 보존) · **CLM/PLU**(개념 관련 layer만 국소 편집) · **W2**(개념만 빼고 직교 성분은 원본 고정). 곧 *locality*: "개념만 바뀌고 나머지는 그대로".

**Q3. FCF의 2-stage 방식과 무엇이 다르고, 어디서 우위인가?**

| | FCF (기존) | LSSE |
|---|---|---|
| 학습 구조 | **2-stage** (explicit→noise, 그다음 implicit) | **단일 루프** (explicit·implicit·retain 동시) |
| forget 목적지 | 임의 **random noise prompt** $p_n$ | **기하학적 null-space** ($c_{dir}$ 성분 0) |
| 학습 범위 | text encoder **전체** | CAP 상위 **top-K layer만** (기본 3) |
| retain 손실 | MSE | InfoNCE (CSR) |

- **학습 속도·효율:** 전체가 아니라 top-K layer만 학습 → trainable 파라미터·메모리 감소, 단일 루프라 스케줄 단순, noise 어휘(vocabulary) 구축 단계 불필요.
- **원리:** 손으로 고른 임의 noise 목적지 대신 데이터에서 뽑은 **개념 축**으로 보내므로 더 원리적이고 재현성이 좋음.
- **성능(로컬 통합 harness, NudeNet v3 · thr 0.3 · 5×50):** Raw 62.0 → vanilla LSSE 46.0 → **+PLU 21.6** → **+PLU+W2 20.8**. 이 harness에서 vanilla LSSE는 FCF-P보다 개선, PLU가 결정적.

> ⚠️ **정직한 단서:** ASR 우열은 **평가 harness에 따라 다릅니다.** 위 수치는 프로젝트 로컬 통합 harness 기준이며, 저자 코드·데이터로 **논문 충실 재현한 FCF-P 자체는 매우 강합니다**(다른 metric/full-set). 세미나에서는 'LSSE의 구조적 이점(단순·고속·국소성)'과 'harness별 ASR'을 구분해 제시하길 권합니다. 또한 LSSE는 text-encoder-only 한계(ASR ≈20 바닥)를 드러냈고, 그것이 DACE→ODACE 연구로 이어졌습니다.""")


code(r"""# === Fig L1: concept null-space projection (CNP) ============================
fig, ax = plt.subplots(figsize=(7.6, 6.6))
cdir = np.array([1.0, 1.0]); cdir /= np.linalg.norm(cdir)
perp = np.array([-cdir[1], cdir[0]])
z = np.array([1.6, 0.35]); proj = np.dot(z, cdir)*cdir; zp = z - proj

ax.plot([-1.7*cdir[0], 1.95*cdir[0]], [-1.7*cdir[1], 1.95*cdir[1]], color=COL["forget"], lw=2.2)
ax.plot([-1.95*perp[0], 1.95*perp[0]], [-1.95*perp[1], 1.95*perp[1]],
        color=COL["prop"], lw=2.2, ls=(0, (5, 3)))
lbl(ax, tuple(cdir*2.05), r"$c_{dir}$", color=COL["forget"], fc="#fdeef0", fs=12)
lbl(ax, (perp[0]*1.75, perp[1]*1.7), "null-space $c_{dir}^{\\perp}$",
    color=COL["prop"], fc="#eaf5f2", fs=11)

ax.plot([z[0], proj[0]], [z[1], proj[1]], color="#c3c9cf", ls=":", lw=1.4)
ax.plot([z[0], zp[0]], [z[1], zp[1]], color="#c3c9cf", ls=":", lw=1.4)
arrow(ax, (0, 0), tuple(z), COL["axis"])
arrow(ax, (0, 0), tuple(proj), COL["forget"])
arrow(ax, (0, 0), tuple(zp), COL["prop"])
node(ax, (0, 0), COL["ink"], s=60)
lbl(ax, (z[0]+0.16, z[1]+0.12), r"$z$", color=COL["axis"], fs=13)
lbl(ax, (proj[0]-0.95, proj[1]+0.18), r"$\mathrm{proj}_{c_{dir}}z\to 0$",
    color=COL["forget"], fc="#fdeef0", fs=11)
lbl(ax, (zp[0]-0.95, zp[1]-0.38), r"$z^{\perp}$ kept", color=COL["prop"],
    fc="#eaf5f2", fs=11)

ax.set_title("CNP zeroes the concept-axis scalar $(z\\cdot c_{dir})$")
ax.set_xlim(-2.2, 2.4); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); clean(ax)
save(fig, "L1_nullspace_projection")
plt.show()""")

code(r"""# === Fig L2: rerouting  vs  W2 orthogonal anchoring ========================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharex=True, sharey=True,
                         constrained_layout=True)
cdir = np.array([1.0, 1.0]) / np.sqrt(2); perp = np.array([-cdir[1], cdir[0]])
z0 = np.dot(np.array([1.5, 0.4]), perp) * perp

for ax, title, anchored in [(axes[0], "Without anchor: concept REROUTES", False),
                            (axes[1], "With $W_2$: $\\perp$ part ANCHORED", True)]:
    ax.plot([-1.9*perp[0], 1.9*perp[0]], [-1.9*perp[1], 1.9*perp[1]],
            color=COL["prop"], lw=2.2, ls=(0, (5, 3)))
    ax.plot([-1.6*cdir[0], 1.9*cdir[0]], [-1.6*cdir[1], 1.9*cdir[1]],
            color=COL["forget"], lw=1.6, alpha=.55)
    node(ax, z0, COL["axis"])
    lbl(ax, (z0[0]+0.12, z0[1]+0.18), r"$z^{\perp}_{frozen}$", fs=11)
    if anchored:
        ax.scatter(*z0, facecolors="none", edgecolors=COL["anchor"], s=420, lw=2.4)
        lbl(ax, (z0[0], z0[1]-0.6), "L2 anchor\n(stays put)", color=COL["anchor"],
            fc="#f2eefb", fs=11)
        ax.set_title(title, color=COL["anchor"])
    else:
        drift = z0 + perp*0.9 - cdir*0.15
        arrow(ax, tuple(z0), tuple(drift), COL["forget"], lw=2.6)
        node(ax, drift, COL["forget"])
        lbl(ax, (drift[0], drift[1]+0.22), "concept\nre-emerges", color=COL["forget"],
            fc="#fdeef0", fs=11)
        ax.set_title(title, color=COL["forget"])
    ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.0, 2.0); ax.set_aspect("equal"); clean(ax)
fig.suptitle("W2 keeps the orthogonal component close to the frozen model",
             fontsize=14, fontweight="bold", color=COL["ink"])
save(fig, "L2_rerouting_vs_anchor")
plt.show()""")

code(r"""# === Fig L3: CNP loss landscape + training decay ===========================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2),
                         gridspec_kw=dict(width_ratios=[1.1, 1]), constrained_layout=True)
ax = axes[0]
gx, gy = np.meshgrid(np.linspace(-2, 2, 240), np.linspace(-2, 2, 240))
cdir = np.array([1, 1]) / np.sqrt(2)
L = (gx*cdir[0] + gy*cdir[1])**2
cs = ax.contourf(gx, gy, L, levels=22, cmap="GnBu")
ax.contour(gx, gy, L, levels=8, colors="white", linewidths=0.4, alpha=.5)
ax.plot([-2*cdir[1], 2*cdir[1]], [2*cdir[0], -2*cdir[0]], color=COL["prop"], lw=2.6,
        ls=(0, (5, 3)))
lbl(ax, (1.0, -1.4), "null-space\n(valley $L=0$)", color=COL["prop"], fc="#eaf5f2", fs=10)
arrow(ax, (1.4, 1.0), (0.35, -0.35), COL["forget"], lw=2.4)
lbl(ax, (1.15, 0.62), r"$-\nabla L$", color=COL["forget"], fc="#fdeef0", fs=12)
fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.02, label=r"$L_{CNP}=(z\cdot c_{dir})^2$")
ax.set_title("CNP loss landscape"); ax.set_aspect("equal"); clean(ax)

ax = axes[1]
ep = np.arange(0, 60)
l, = ax.plot(ep, np.exp(-ep/12), color=COL["prop"], lw=3.4); glow(l)
ax.fill_between(ep, np.exp(-ep/12), color=COL["prop"], alpha=.12)
ax.set_xlabel("epoch"); ax.set_ylabel(r"$\mathbb{E}\,|z\cdot c_{dir}|$")
ax.set_title("Concept-axis scalar $\\to 0$")
ax.set_xlim(0, 59); ax.set_ylim(0, 1.05)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.grid(axis="y", color=COL["grid"], lw=1)
save(fig, "L3_cnp_loss")
plt.show()""")

code(r"""# === Fig L4: PLU — progressive layer unlocking =============================
fig, ax = plt.subplots(figsize=(9, 4.9))
n_layers, n_epochs = 12, 60
sched = np.zeros((n_layers, n_epochs))
for e in range(n_epochs):
    sched[:min(n_layers, 1 + e // 6), e] = 1
ax.imshow(sched, aspect="auto", cmap="BuGn", origin="lower",
          extent=[0, n_epochs, 0, n_layers], vmin=0, vmax=1.35)
ks = [min(n_layers, 1 + e // 6) for e in range(n_epochs)]
ax.step(range(n_epochs), ks, color=COL["forget"], lw=2.6, where="post")
ax.set_xlabel("epoch"); ax.set_ylabel("text-encoder layer (top-K first)")
ax.set_title("PLU — unlock 1 layer, then progressively more (CAP-ranked)")
lbl(ax, (10, 10.4), "frozen", color=COL["muted"], fs=11)
lbl(ax, (47, 2.2), "trainable", color=COL["prop"], fc="#eaf5f2", fs=11)
save(fig, "L4_progressive_layer_unlocking")
plt.show()""")


md(r"""## Summary — three verbs, one goal""")

code(r"""# === Fig SUM: ODACE (bend) · Sph+OT (move) · LSSE (project) ================
fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

# ODACE
ax = axes[0]
e0 = np.array([0, 0]); ep = np.array([1.2, 0.9]); et = e0 - 0.6*(ep-e0)
arrow(ax, tuple(e0), tuple(ep), COL["forget"]); arrow(ax, tuple(e0), tuple(et), COL["prop"])
node(ax, e0, COL["axis"], s=55)
ax.set_title("ODACE — bend the output", color=COL["prop"])
lbl(ax, (0, -1.55), r"$\epsilon_{target}=\epsilon_0-\eta(\epsilon_p-\epsilon_0)$", fs=11)
ax.set_xlim(-1.4, 1.6); ax.set_ylim(-2.1, 1.4)

# Sph+OT
ax = axes[1]
t = np.linspace(0, np.pi, 200); ax.plot(np.cos(t), np.sin(t), color=COL["axis"], lw=1.8)
a0, a1 = 0.45, 1.4
p_x = np.array([np.cos(a1), np.sin(a1)]); p_c = np.array([np.cos(a0), np.sin(a0)])
arc = np.array([exp_map(p_x, -s*0.7*log_map(p_x, p_c)) for s in np.linspace(0, 1, 30)])
g, = ax.plot(arc[:, 0], arc[:, 1], color=COL["prop"], lw=3.2); glow(g)
node(ax, p_x, COL["axis"]); node(ax, p_c, COL["forget"])
ax.set_title("Sph+OT — move on the sphere", color=COL["prop"])
lbl(ax, (0, -1.55), r"$\mathrm{Exp}_{u_x}(-\mu_p\,\mathrm{Log}_{u_x}(u_c))$", fs=11)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-2.1, 1.4)

# LSSE
ax = axes[2]
cd = np.array([1, 1])/np.sqrt(2); pp = np.array([-cd[1], cd[0]])
z = np.array([1.3, 0.3]); zp = z - np.dot(z, cd)*cd
ax.plot([-1.3*cd[0], 1.5*cd[0]], [-1.3*cd[1], 1.5*cd[1]], color=COL["forget"], lw=2)
ax.plot([-1.5*pp[0], 1.5*pp[0]], [-1.5*pp[1], 1.5*pp[1]], color=COL["prop"], lw=2, ls=(0, (5, 3)))
arrow(ax, (0, 0), tuple(z), COL["axis"]); arrow(ax, (0, 0), tuple(zp), COL["prop"])
ax.set_title("LSSE — project the axis out", color=COL["prop"])
lbl(ax, (0, -1.55), r"$(z\cdot c_{dir})^2\to0$  +  $\perp$ anchor", fs=11)
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.1, 1.4)

for ax in axes:
    ax.set_aspect("equal"); clean(ax)
fig.suptitle("Three spaces · three verbs · one goal (erase concept, keep utility)",
             fontsize=15, fontweight="bold", color=COL["ink"])
save(fig, "SUM_comparison")
plt.show()""")

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

code(r"""# === Anim A1: ODACE — the trajectory bends away from the concept ===========
fig, ax = plt.subplots(figsize=(7.2, 5.9))
forget = np.array([1.4, -1.2]); safe = np.array([-1.4, -1.2]); start = np.array([0.0, 1.85])

def _bez(p0, p1, p2, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return (1-t)**2*p0 + 2*(1-t)*t*p1 + t**2*p2

base = _bez(start, np.array([0.95, 0.3]), forget)
od = _bez(start, np.array([-0.95, 0.3]), safe)
for c, col, lab in [(forget, COL["forget"], "forget mode"), (safe, COL["safe"], "safe mode")]:
    ax.add_patch(plt.Circle(tuple(c), 0.6, color=col, alpha=.14, zorder=0)); node(ax, c, col)
    ax.text(c[0], c[1]-0.95, lab, ha="center", color=col, fontsize=11)
node(ax, start, COL["axis"]); lbl(ax, (start[0]+0.32, start[1]), r"$z_T$", fs=12)
lB, = ax.plot([], [], color=COL["base"], lw=3.4, label="baseline"); glow(lB)
lO, = ax.plot([], [], color=COL["prop"], lw=3.4, label="ODACE"); glow(lO)
ax.legend(loc="upper right")
ax.set_title("ODACE bends the trajectory at its source")
ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.3, 2.35); ax.set_aspect("equal"); clean(ax)

N = len(base); F = 60
def _upd(f):
    k = max(2, int(N*(f+1)/F))
    lB.set_data(base[:k, 0], base[:k, 1]); lO.set_data(od[:k, 0], od[:k, 1])
    return lB, lO

anim = FuncAnimation(fig, _upd, frames=F, interval=50, blit=False)
p = save_gif(anim, "A1_odace_trajectory", fps=20)
IPyImage(filename=p)""")

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

code(r"""# === Anim A3: LSSE — the concept-axis component shrinks to 0 ================
fig, ax = plt.subplots(figsize=(6.9, 6.4))
cdir = np.array([1, 1]) / np.sqrt(2); perp = np.array([-cdir[1], cdir[0]])
ax.plot([-1.7*cdir[0], 1.95*cdir[0]], [-1.7*cdir[1], 1.95*cdir[1]], color=COL["forget"], lw=2)
ax.plot([-1.95*perp[0], 1.95*perp[0]], [-1.95*perp[1], 1.95*perp[1]], color=COL["prop"],
        lw=2, ls=(0, (5, 3)))
lbl(ax, tuple(cdir*2.05), r"$c_{dir}$", color=COL["forget"], fc="#fdeef0", fs=12)
lbl(ax, (perp[0]*1.7, perp[1]*1.62), "null-space", color=COL["prop"], fc="#eaf5f2", fs=11)
z = np.array([1.6, 0.35]); proj0 = np.dot(z, cdir)*cdir; zp = z - proj0
node(ax, (0, 0), COL["ink"], s=55); node(ax, tuple(zp), COL["prop"])
zln, = ax.plot([], [], color=COL["axis"], lw=2.8)
zhd = ax.scatter([z[0]], [z[1]], s=80, color=COL["axis"], edgecolors="white", zorder=9)
pln, = ax.plot([], [], color=COL["forget"], lw=3)
txt = ax.text(0, -1.95, "", ha="center", fontsize=13, color=COL["forget"])
ax.set_title("CNP drives $(z\\cdot c_{dir})$ to 0 — $z$ lands in the null-space")
ax.set_xlim(-2.2, 2.4); ax.set_ylim(-2.2, 2.2); ax.set_aspect("equal"); clean(ax)

NF = 50; s0 = np.dot(z, cdir)
def _upd(f):
    s = 1 - f/(NF-1)                 # 1 -> 0
    cur = zp + s*proj0
    zln.set_data([0, cur[0]], [0, cur[1]]); zhd.set_offsets([cur])
    pln.set_data([zp[0], cur[0]], [zp[1], cur[1]])
    txt.set_text(r"$z\cdot c_{dir}$ = %.2f" % (s*s0))
    return zln, zhd, pln, txt

anim = FuncAnimation(fig, _upd, frames=NF, interval=60, blit=False)
p = save_gif(anim, "A3_lsse_projection", fps=18)
IPyImage(filename=p)""")

code(r"""# === Bundle every figure for download (Colab) ==============================
import shutil
shutil.make_archive("method_figures", "zip", "figures")
print("zipped -> method_figures.zip")
try:
    from google.colab import files  # type: ignore
    files.download("method_figures.zip")
except Exception as e:
    print("(not in Colab) figures are in ./figures/ —", e)""")


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
