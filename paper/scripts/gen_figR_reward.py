"""Training-reward grid (raw faint + smoothed bold), same panels as Figure 1 (b)-(h)."""
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
src = open('paper/scripts/gen_story_figs.py').read(); exec(src[:src.index('# ---------------- F1')])
g = open('paper/scripts/gen_fig1_grid.py').read(); exec(g[g.index('panels = ['):g.index(']\n', g.index('panels = [')) + 2])
names = sorted({n for _, a, b, o in panels for n in a + b})
ed = events(names)
fig, axs = plt.subplots(2, 4, figsize=(13, 5.6)); axs = axs.ravel()
def draw(ax, arms, col, lab):
    sm = []
    for n in arms:
        x, y = series(ed, n, 'train_acc', roll=0)
        if len(x) == 0: continue
        ax.plot(x, y, color=col, lw=0.6, alpha=0.22)
        xs, ys = series(ed, n, 'train_acc', roll=10); sm.append(dict(zip(xs, ys)))
    if sm:
        steps = sorted(set.union(*[set(d) for d in sm])); M = np.array([[d.get(s, np.nan) for s in steps] for d in sm])
        ax.plot(steps, np.nanmean(M, 0), color=col, lw=2, label=lab)
for ax, (title, near, cast, _) in zip(axs, panels):
    draw(ax, near, RED, 'round-to-nearest'); draw(ax, cast, BLUE, 'Fair Cast')
    ax.set_title(title, fontsize=8.5, loc='left')
    ax.set_xlabel('optimizer step'); ax.grid(alpha=0.25)
for ax in axs[::4]: ax.set_ylabel('training reward (mean over batch)')
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, ['round-to-nearest export (bold: 10-step moving average, faint: raw)', 'Fair Cast'], ncol=2, loc='lower center', frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig('paper/figures/figR_reward.png', dpi=190); plt.close(fig); print('ok', names)
