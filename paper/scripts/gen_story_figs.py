"""Story figures for the paper: F1 problem, F2 zebra headline, F3 breadth grid, F4 checkpoint!=model, F5 failures+fix.
Data: results/evals.json (held-out evals) + per-step gen/train events parsed from results/logs/*.log (train_acc, mean_len, frac_bf16_changed, log-ratios)."""
import json, glob, os, statistics as st, collections
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
E = json.load(open('results/evals.json'))
# MBPP+ shares 154/378 task ids with the MBPP training split: plot the held-out-224 curves (results/mbppplus_heldout_curves.json, from per-problem eval_hits)
if os.path.exists('results/mbppplus_heldout_curves.json'):
    for _n, _c in json.load(open('results/mbppplus_heldout_curves.json')).items():
        if _n in E: E[_n]['eval'] = [[int(k), v] for k, v in sorted(_c['heldout'].items(), key=lambda kv: int(kv[0]))]
RED, BLUE, GREY, GREEN = '#c0392b', '#2471a3', '#555555', '#1e8449'
plt.rcParams.update({'font.size': 8.5, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlesize': 9, 'legend.fontsize': 7.5})

def events(names):
    """per-arm dict of step->gen/train rows, from every log that mentions the arm"""
    out = {n: {'gen': {}, 'train': {}} for n in names}
    logs = set(E[n]['log'] for n in names if n in E) | set(glob.glob('results/logs/smoke*.log'))
    for lg in logs:
        p = lg if os.path.exists(lg) else os.path.join('results/logs', lg)
        if not os.path.exists(p): continue
        for l in open(p, errors='replace'):
            if '"event": "gen"' not in l and '"event": "train"' not in l: continue
            try: r = json.loads(l[l.index('{'):])
            except Exception: continue
            n = r.get('name')
            if n in out: out[n][r['event']].setdefault(r['step'], r)
    return out

def ev(n): return np.array(sorted(E[n]['eval'])) if n in E and E[n]['eval'] else np.zeros((0, 2))
def series(ed, n, key, kind='gen', roll=10, drop=None):
    d = ed[n][kind]; xs = sorted(d); ys = [d[s].get(key) for s in xs]
    pts = [(x, y) for x, y in zip(xs, ys) if y is not None and not (drop and drop(x))]
    if not pts: return np.zeros(0), np.zeros(0)
    xs, ys = map(np.array, zip(*pts))
    if roll and len(ys) >= roll: ys = np.convolve(ys, np.ones(roll) / roll, mode='valid'); xs = xs[roll - 1:]
    return xs, ys
def acc(ax, n, col, lab=None, ls='-', lw=1.6):
    e = ev(n)
    if len(e): ax.plot(e[:, 0], e[:, 1], ls, color=col, lw=lw, label=lab, marker='o' if len(e) < 20 else None, ms=2.5)
def band(ax, names, col, lab):
    es = [ev(n) for n in names if len(ev(n))]
    steps = sorted(set.intersection(*[set(e[:, 0]) for e in es]))
    M = np.array([[dict(e.tolist())[s] for s in steps] for e in es])
    mu, sd = M.mean(0), M.std(0, ddof=1) if len(M) > 1 else np.zeros(M.shape[1])
    ax.plot(steps, mu, color=col, lw=2, label=lab); ax.fill_between(steps, mu - sd, mu + sd, color=col, alpha=0.3, lw=0)

# ---------------- F1: the problem ----------------
ed = events(['smoke_fp32sampler', 'smoke_cast_kk', 'p16_zebra_cast', 'p16_zebra_srste', 'p16_kk_cast', 'p16_kk_srste'])
fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.3))
# (a) smoke: eval acc after 0/1/2 steps (first duplicate step-0 row is the pre-step eval)
def smoke(n):
    rows = [json.loads(l[l.index('{'):]) for f in glob.glob('results/logs/smoke*.log') for l in open(f) if f'"name": "{n}"' in l and '"event": "eval"' in l]
    return [r['eval_acc'] for r in rows][:4]
s32, scast = smoke('smoke_fp32sampler'), smoke('smoke_cast_kk')
x = np.arange(len(s32)); w = 0.38
ax[0].bar(x - w / 2, scast, w, color=RED, label='bf16 export (nearest)'); ax[0].bar(x + w / 2, s32, w, color=GREEN, label='fp32 sampler')
ax[0].set_xticks(x); ax[0].set_xticklabels(['before\nstep 1', 'after\nstep 1', 'after\nstep 2', 'after\nstep 3'][:len(x)]); ax[0].set_ylim(0, 1); ax[0].set_ylabel('held-out accuracy (64 problems)')
ax[0].set_title('(a) Single-step test', loc='left', fontsize=9); ax[0].legend(loc='upper left', frameon=False)
# (b) fraction of served weights changed per sync
# the eval-time re-sync resets the change counter, so step%10==0 is logged as 0 for nearest: dropped
for n, col, lab in (('p16_zebra_cast', RED, 'round-to-nearest'), ('p16_zebra_srste', BLUE, 'Fair Cast (stochastic rounding)')):
    xs, ys = series(ed, n, 'frac_bf16_changed', roll=3, drop=lambda x: x % 10 == 0); ax[1].plot(xs, 100 * np.maximum(ys, 1e-3), color=col, lw=1.4, label=lab)
ax[1].set_yscale('log'); ax[1].set_xlabel('optimizer step'); ax[1].set_ylabel('% of served bf16 weights changed per step'); ax[1].set_title('(b) Served weights changed per step', loc='left', fontsize=9); ax[1].legend(frameon=False, loc='center right', bbox_to_anchor=(1, 0.45))
# (c) hidden vs visible log-ratio
xs, ys = series(ed, 'p16_kk_cast', 'mean_abs_logratio', 'train', roll=5); ax[2].plot(xs, ys, color=RED, lw=1.6, label='nearest: master vs served (hidden)')
xs, ys = series(ed, 'p16_kk_cast', 'loss_side_mean_abs_logratio', 'train', roll=5); ax[2].plot(xs, ys, color=GREY, lw=1.6, ls='--', label='nearest: trainer vs sampler (visible)')
xs, ys = series(ed, 'p16_kk_srste', 'mean_abs_logratio', 'train', roll=5); ax[2].plot(xs, ys, color=BLUE, lw=1.6, label='Fair Cast: master vs served')
ax[2].set_xlabel('optimizer step'); ax[2].set_ylabel('mean per-token |log-ratio| (nats)'); ax[2].set_title('(c) Mean per-token |log ρ|', loc='left', fontsize=9); ax[2].legend(frameon=False, loc='center right'); ax[2].set_ylim(bottom=0)
fig.tight_layout(w_pad=2.5); fig.savefig('paper/figures/figS1_problem.png', dpi=170); plt.close(fig)

# ---------------- F2: zebra headline ----------------
near = ['p16_zebra_cast', 'p16_zebra_cast_s1', 'p30_zebra_cast_s2']; fair = ['p16_zebra_srste', 'p16_zebra_srste_s1', 'p30_zebra_sr_s2', 'p25_zebra_srste_s0']
ed = events(near + fair + ['p23_zebra_fp32sampler'])
fig, ax = plt.subplots(3, 1, figsize=(6.4, 7.2), sharex=True, gridspec_kw={'height_ratios': [2.2, 1, 1]})
band(ax[0], near, RED, 'round-to-nearest export (3 seeds, mean ± 1 s.d.)'); band(ax[0], fair, BLUE, 'Fair Cast (3 seeds, mean ± 1 s.d.)')
acc(ax[0], 'p23_zebra_fp32sampler', GREEN, 'fp32-sampler oracle (no rounding, seed 0)', ls='--', lw=1.4)
ax[0].set_ylabel('held-out accuracy (512 puzzles)'); ax[0].legend(frameon=False, loc='lower left'); ax[0].set_ylim(0, 0.8); ax[0].set_title('(a) Held-out accuracy', loc='left', fontsize=9)
for grp, col in ((near, RED), (fair, BLUE)):
    for i, n in enumerate(grp):
        xs, ys = series(ed, n, 'train_acc'); ax[1].plot(xs, ys, color=col, lw=1.1, alpha=0.85)
        xs, ys = series(ed, n, 'mean_len'); ax[2].plot(xs, ys, color=col, lw=1.1, alpha=0.85)
ax[2].axhline(2048, color='k', lw=0.8, ls=':'); ax[2].text(2, 2048 * 0.93, 'token cap', fontsize=7)
ax[1].set_ylabel('training reward\n(10-step mean)'); ax[2].set_ylabel('response length\n(tokens, 10-step mean)'); ax[2].set_xlabel('optimizer step')
fig.tight_layout(); fig.savefig('paper/figures/figS2_zebra.png', dpi=170); plt.close(fig)

# ---------------- F3: breadth grid ----------------
panels = [
    ('Knights & knaves · 1.7B · AdamW', ['p16_kk_cast'], ['p16_kk_srste'], 'p22_kk_fp32sampler'),
    ('Knights & knaves · 1.7B · Muon (2 seeds)', ['p21_kk_muon_cast', 'p30_muon_kk_cast_s1'], ['p21_kk_muon_srste', 'p30_muon_kk_sr_s1'], None),
    ('Knights & knaves · 0.6B · AdamW', ['p24_kk_q06_cast'], ['p24_kk_q06_srste'], None),
    ('Knights & knaves · 4B · AdamW', ['p24_kk_q4b_cast_h200'], ['p24_kk_q4b_srste_h200'], None),
    ('Nim (6k tokens) · 1.7B · AdamW', ['p29b_nim_cast_s0'], ['p29b_nim_sr_s0'], None),
    ('GSM8K · 0.6B · AdamW (3 seeds)', ['p32_q06_gsm8k_cast_s0', 'p32f_q06_gsm8k_cast_s1', 'p32g_q06_gsm8k_cast_s2'], ['p32e_q06_gsm8k_srste_s0', 'p32f_q06_gsm8k_srste_s1', 'p32g_q06_gsm8k_srste_s2'], None),
    ('MATH L3–5 · 0.6B · AdamW (3 seeds)', ['p32_q06_math_cast_s0', 'p32f_q06_math_cast_s1', 'p32g_q06_math_cast_s2'], ['p32e_q06_math_srste_s0', 'p32f_q06_math_srste_s1', 'p32g_q06_math_srste_s2'], None),
]
fig, axs = plt.subplots(2, 4, figsize=(12.5, 5.6)); axs = axs.ravel()
for a, (title, n0s, n1s, orc) in zip(axs, panels):
    if len(n0s) == 1: acc(a, n0s[0], RED); acc(a, n1s[0], BLUE)
    else: band(a, n0s, RED, None); band(a, n1s, BLUE, None)
    if orc: acc(a, orc, GREEN, ls='--', lw=1.2)
    a.set_title(title, fontsize=9); a.grid(alpha=0.25)
    f0 = np.mean([ev(n)[-1, 1] for n in n0s]); f1 = np.mean([ev(n)[-1, 1] for n in n1s])
    a.text(0.98, 0.04, f"final {f0:.2f} → {f1:.2f}", transform=a.transAxes, ha='right', fontsize=7.5, color='#333')
axs[-1].axis('off')
for a in axs[4:7]: a.set_xlabel('optimizer step')
for a in axs[::4]: a.set_ylabel('held-out accuracy')
h = [plt.Line2D([], [], color=RED, lw=1.6), plt.Line2D([], [], color=BLUE, lw=1.6), plt.Line2D([], [], color=GREEN, lw=1.2, ls='--')]
axs[-1].legend(h, ['round-to-nearest export', 'Fair Cast (stochastic rounding\n+ train on served copy)', 'fp32-sampler oracle'], loc='center', frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig('paper/figures/figS3_breadth.png', dpi=170); plt.close(fig)

# ---------------- F4: checkpoint != model ----------------
rows = []
for n, a in E.items():
    if not a.get('master') or not a.get('eval') or (a.get('n_eval') or 0) < 164: continue
    evd = dict(a['eval'])
    for s, m, mb in a['master']:
        if s in evd and m is not None and s > 0: rows.append((n, a.get('sync'), s, evd[s], m))
worst = {}
for n, sy, s, served, master in rows:
    g = served - master
    if n not in worst or g > worst[n][3]: worst[n] = (sy, s, served, g, master)
sel = sorted(worst.items(), key=lambda kv: (kv[1][0] != 'cast', -kv[1][3]))
sel = [kv for kv in sel if kv[1][0] == 'cast'][:9] + [kv for kv in sel if kv[1][0] != 'cast'][:6]
fig, ax = plt.subplots(figsize=(10, 3.8)); x = np.arange(len(sel)); w = 0.4
def pretty(n):
    t = 'HumanEval+' if 'heplus' in n else 'MBPP+'; sd = n.split('_s')[-1].split('_')[0]; return f'{t} seed {sd}'
ax.bar(x - w / 2, [100 * v[2] for _, v in sel], w, color=[RED if v[0] == 'cast' else BLUE for _, v in sel], label='served bf16 copy')
ax.bar(x + w / 2, [100 * v[4] for _, v in sel], w, color='white', edgecolor=[RED if v[0] == 'cast' else BLUE for _, v in sel], hatch='////', label='fp32 master checkpoint')
ax.set_xticks(x); ax.set_xticklabels([f'{pretty(n)}\nstep {v[1]}' for n, v in sel], rotation=40, ha='right', fontsize=7)
ax.set_ylabel('held-out accuracy (%)'); ax.set_title('')
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.42), ncol=2); ax.set_ylim(0, 72); fig.tight_layout(); fig.savefig('paper/figures/figS4_ckpt.png', dpi=170, bbox_inches='tight'); plt.close(fig)
print('F4 arms:', [(n, v[0], v[1], round(v[2], 3), round(v[4], 3)) for n, v in sel])

# ---------------- F5: lag as a tunable knob (list functions) ----------------
names = ['p20_list_functions_cast_s0', 'p20_list_functions_sr_s0', 'p26_lf_sr_ema05', 'p27_lf_sr_ema05_s1', 'p26_lf_sr_ema01']
ed = events(names)
fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
acc(ax[0], 'p20_list_functions_cast_s0', RED, 'round-to-nearest (implicit lag)'); acc(ax[0], 'p20_list_functions_sr_s0', BLUE, 'Fair Cast, no lag')
acc(ax[0], 'p26_lf_sr_ema05', GREEN, 'Fair Cast + EMA lag α=0.05, seed 0'); acc(ax[0], 'p27_lf_sr_ema05_s1', GREEN, 'Fair Cast + EMA lag α=0.05, seed 1', ls='--'); acc(ax[0], 'p26_lf_sr_ema01', '#7d3c98', 'Fair Cast + EMA lag α=0.01', ls=':')
ax[0].set_title('(a) Held-out accuracy', loc='left'); ax[0].set_ylabel('held-out accuracy'); ax[0].set_xlabel('optimizer step'); ax[0].legend(frameon=False, loc='lower right', fontsize=7.5); ax[0].grid(alpha=0.25)
for n, col, ls in (('p20_list_functions_cast_s0', RED, '-'), ('p20_list_functions_sr_s0', BLUE, '-'), ('p26_lf_sr_ema05', GREEN, '-'), ('p27_lf_sr_ema05_s1', GREEN, '--')):
    xs, ys = series(ed, n, 'mean_len'); ax[1].plot(xs, ys, color=col, ls=ls, lw=1.3)
ax[1].set_title('(b) Training response length', loc='left'); ax[1].set_ylabel('tokens (10-step mean)'); ax[1].set_xlabel('optimizer step'); ax[1].grid(alpha=0.25)
fig.tight_layout(); fig.savefig('paper/figures/figS5_lag.png', dpi=170); plt.close(fig)
