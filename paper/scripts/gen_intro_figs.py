"""Programmatic schematics for the introduction: (a) a weight on the bf16 grid under nearest vs stochastic rounding; (b) the training pipeline with the visible and hidden ratios."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
RED, BLUE, GREY, GREEN = '#c0392b', '#2471a3', '#7f8c8d', '#1e8449'
rng = np.random.default_rng(3)
# ---- (a) grid walk ----
u = 1.0                      # one bf16 spacing (ULP) in these units
T = 60; eta = 0.012 * u      # per-step master movement, ~1% of a ULP (η=1e-6 regime)
theta = 0.3 * u + np.cumsum(eta * (1 + 0.3 * rng.standard_normal(T)))
near = np.round(theta / u) * u
frac = theta / u - np.floor(theta / u); sr = (np.floor(theta / u) + (rng.random(T) < frac)) * u
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
for a, served, col, title in ((ax[0], near, RED, 'Round-to-nearest export'), (ax[1], sr, BLUE, 'Stochastic-rounding export (Fair Cast)')):
    for g in np.arange(0, 2.01, 1.0): a.axhline(g, color='#bbb', lw=0.8, ls=':')
    a.axhspan(0.0, 0.5, color='#f3f3f3', zorder=0); a.text(T - 1, 0.25, 'dead zone: updates\nbelow u/2 are discarded', ha='right', va='center', fontsize=7.5, color=GREY) if col == RED else None
    a.plot(theta, color='k', lw=1.6, label='fp32 master θ')
    a.step(np.arange(T), served, where='post', color=col, lw=1.8, label='served bf16 copy q')
    a.set_title(title, loc='left', fontsize=10); a.set_xlabel('optimizer step'); a.set_yticks([0, 1, 2]); a.set_yticklabels(['w₀', 'w₀ + u', 'w₀ + 2u']); a.grid(False)
    for s in ('top', 'right'): a.spines[s].set_visible(False)
ax[0].set_ylabel('weight value (bf16 grid spacing u ≈ |w|/128)'); ax[0].legend(frameon=False, loc='upper left', fontsize=8); ax[1].legend(frameon=False, loc='upper left', fontsize=8)
ax[0].annotate('served value frozen\nwhile master moves', xy=(25, 1.0), xytext=(8, 1.55), fontsize=8, color=RED, arrowprops=dict(arrowstyle='->', color=RED, lw=0.8))
ax[1].annotate('flips with probability\n(θ − ⌊θ⌋)/u: E[q] = θ', xy=(30, 1.0), xytext=(4, 1.55), fontsize=8, color=BLUE, arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
fig.tight_layout(); fig.savefig('paper/figures/figI1_grid.png', dpi=200); plt.close(fig)
# ---- (b) pipeline ----
fig, ax = plt.subplots(figsize=(10.5, 3.6)); ax.set_xlim(0, 10.2); ax.set_ylim(-0.2, 3.6); ax.axis('off')
def box(x, y, w, h, text, col='#fff', ec='#444', fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.12', fc=col, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, fontweight='bold' if bold else 'normal')
def arrow(x0, y0, x1, y1, col='#444', ls='-', lw=1.3, text=None, tpos=None, tcol=None, fs=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=12, color=col, lw=lw, ls=ls))
    if text: ax.text(*(tpos or ((x0 + x1) / 2, (y0 + y1) / 2 + 0.15)), text, ha='center', va='bottom', fontsize=fs, color=tcol or col)
box(0.3, 2.2, 1.9, 0.9, 'fp32 master θ\n(optimizer state)', '#f7f7f7', bold=True)
box(3.3, 2.2, 2.0, 0.9, 'export\nround θ → bf16', '#fdebd0', ec='#d35400')
box(6.3, 2.2, 1.9, 0.9, 'served copy q\n(vLLM / SGLang)', '#f7f7f7', bold=True)
box(6.3, 0.4, 1.9, 0.9, 'rollouts y ~ π_q\nrewards r', '#eaf2f8')
box(3.3, 0.4, 2.0, 0.9, 'trainer forward\n(bf16 copy of θ ≈ q)', '#fdebd0', ec='#d35400')
box(0.3, 0.4, 1.9, 0.9, 'policy-gradient\nupdate of θ', '#f7f7f7')
arrow(2.2, 2.65, 3.3, 2.65); arrow(5.3, 2.65, 6.3, 2.65); arrow(7.25, 2.2, 7.25, 1.3); arrow(6.3, 0.85, 5.3, 0.85); arrow(3.3, 0.85, 2.2, 0.85); arrow(1.25, 1.3, 1.25, 2.2)
ax.annotate('', xy=(5.3, 1.05), xytext=(6.3, 2.35), arrowprops=dict(arrowstyle='<->', color=GREEN, lw=1.4, ls='--'))
ax.text(8.35, 1.75, 'visible ratio\nπ_trainer/π_sampler\nchecked by PPO clip,\nTIS, IcePop: ≈ 1', fontsize=8, color=GREEN, ha='left', va='center')
ax.annotate('', xy=(6.3, 2.95), xytext=(2.2, 2.95), arrowprops=dict(arrowstyle='<->', color=RED, lw=1.6, ls='--'))
ax.text(4.3, 3.3, 'hidden ratio π_θ/π_q: never computed; grows to |log ρ| ≈ 0.1–0.45 under round-to-nearest', fontsize=8.5, color=RED, ha='center', va='center')
ax.text(4.3, 0.05, 'Fair Cast changes the two shaded boxes: unbiased rounding at the export, and a trainer forward on the exact served copy q', fontsize=8.5, color='#d35400', ha='center', va='center', style='italic')
fig.savefig('paper/figures/figI2_pipeline.png', dpi=200, bbox_inches='tight'); plt.close(fig)
print('ok')
