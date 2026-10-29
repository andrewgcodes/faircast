"""Figure 1 for Paper 1: 3x3 results grid on page 1 (FP16-paper style). Reuses the data helpers of gen_story_figs.py."""
import os, re, json, glob
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
src = open('paper/scripts/gen_story_figs.py').read()
exec(src[:src.index('# ---------------- F1')])          # E, events, ev, acc, band, colors
def smoke(n):
    rows = [json.loads(l[l.index('{'):]) for f in glob.glob('results/logs/smoke*.log') for l in open(f) if f'"name": "{n}"' in l and '"event": "eval"' in l]
    return [r['eval_acc'] for r in rows]
panels = [
    ('(a) Zebra puzzles, 1.7B (3 seeds)', ['p16_zebra_cast', 'p16_zebra_cast_s1', 'p30_zebra_cast_s2'], ['p16_zebra_srste', 'p16_zebra_srste_s1', 'p30_zebra_sr_s2'], 'p23_zebra_fp32sampler'),
    ('(b) Knights-and-knaves, 1.7B', ['p16_kk_cast'], ['p16_kk_srste'], 'p22_kk_fp32sampler'),
    ('(c) Knights-and-knaves, Muon (2 seeds)', ['p21_kk_muon_cast', 'p30_muon_kk_cast_s1'], ['p21_kk_muon_srste', 'p30_muon_kk_sr_s1'], None),
    ('(d) Knights-and-knaves, 0.6B', ['p24_kk_q06_cast'], ['p24_kk_q06_srste'], None),
    ('(e) Nim, 1.7B', ['p29b_nim_cast_s0'], ['p29b_nim_sr_s0'], None),
    ('(f) GSM8K, 0.6B (2 seeds)', ['p32_q06_gsm8k_cast_s0', 'p32f_q06_gsm8k_cast_s1'], ['p32e_q06_gsm8k_srste_s0', 'p32f_q06_gsm8k_srste_s1'], None),
    ('(g) MATH levels 3-5, 0.6B (2 seeds)', ['p32_q06_math_cast_s0', 'p32f_q06_math_cast_s1'], ['p32e_q06_math_srste_s0', 'p32f_q06_math_srste_s1'], None),
    ('(h) MBPP+, 1.7B (4 seeds)', ['p6_mbpp_cast', 'p18_mbpp_cast_s1_e512', 'p18_mbpp_cast_s2_e512', 'p18_mbpp_cast_s3_e512'], ['p6_mbpp_srste', 'p18_mbpp_sr_s1_e512', 'p18_mbpp_sr_s2_e512', 'p18_mbpp_sr_s3_e512'], None),
]
# verify arm names exist; fall back to whatever zebra arms are in E
for _, a, b, o in panels:
    for n in a + b + ([o] if o else []):
        if n not in E: print('MISSING', n)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10.5, 'axes.labelsize': 10, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 10})
fig, axs = plt.subplots(2, 4, figsize=(12.5, 5.8)); axs = axs.ravel()
for a, (title, n0s, n1s, orc) in zip(axs, panels):
    if len(n0s) == 1: acc(a, n0s[0], RED); acc(a, n1s[0], BLUE)
    else: band(a, n0s, RED, None); band(a, n1s, BLUE, None)
    if orc and orc in E: acc(a, orc, GREEN, ls='--', lw=1.2)
    a.set_title(title, loc='left'); a.grid(alpha=0.25)
for a in axs[4:]: a.set_xlabel('optimizer step')
for a in axs[::4]: a.set_ylabel('held-out accuracy')
h = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Line2D([], [], color=GREEN, lw=1.4, ls='--')]
fig.legend(h, ['round-to-nearest export (current default)', 'Fair Cast (unbiased export + train on served copy)', 'fp32 sampler (no rounding)'], ncol=3, loc='lower center', frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig('paper/figures/fig1_grid.png', dpi=190); plt.close(fig); print('ok')
