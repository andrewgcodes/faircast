# Fair Cast

Code for **"bf16 Rounding Silently Discards Most RL Policy Updates: A Hidden Training-Inference Mismatch and an Unbiased One-Line Fix"** (Andrew Gao, William Gao; Stanford University).

RL post-training of language models keeps fp32 master weights in the optimizer and serves rollouts from a bf16 copy. Every open-source framework we looked at makes that copy with round-to-nearest, which discards any update smaller than half a bf16 spacing (about |w|/256). At the usual learning rates (around 1e-6) that is almost every update: on Qwen3-1.7B only 0.4–0.5% of served weights change per step and the sampler lags the optimizer by 50–150 steps. The mismatch checks frameworks run (PPO clip, TIS, IcePop) cannot see this, because they compare two copies that sit downstream of the same rounding.

**Fair Cast** is two changes to an existing loop:

1. **Export with an unbiased rounding rule.** Stochastic rounding (`sr_cast_bf16`, E[served] = master, no state) or error feedback (`ef_cast_bf16`, one fp32 residual per parameter) instead of `master.to(torch.bfloat16)`.
2. **Compute the loss on the served copy.** The trainer's forward pass uses the exact bf16 weights that generated the rollouts; gradients reach the fp32 master through a straight-through estimator.

Both halves are needed (paper §5, §7.1). The change is confined to the weight-export call, adds no hyperparameters and costs 22–38 ms per export of 1.7B parameters.

## Install

```bash
pip install torch
pip install -e .            # the faircast package (torch is the only dependency)
python -m faircast          # CPU self-test: unbiasedness, dead zone, error feedback
python -m pytest -q tests   # unit tests
```

## The one-line change

```python
import torch
from faircast import sr_cast_bf16, ef_cast_bf16, sync_to_sampler, ste_params, served_forward

# (1) export. Replace `served.copy_(master)` / `master.to(torch.bfloat16)` in your weight-sync hook:
q = sr_cast_bf16(master_fp32, gen)                 # stochastic rounding, E[q] == master
# or, deterministic, with one fp32 residual tensor per parameter:
q = ef_cast_bf16(master_fp32, resid)               # q_t = round(theta_t + r_{t-1}); r_t = theta_t + r_{t-1} - q_t

# whole model at once (dicts name -> tensor):
sync_to_sampler(master_params, served_params, mode="sr", gen=gen)

# (2) train on the served copy. Either build straight-through params for torch.func.functional_call ...
params = ste_params(model, served)                 # == served in value, d/dtheta == I
logits = torch.func.functional_call(model, params, (), dict(input_ids=ids)).logits
# ... or swap the served values into the fp32 containers for the forward/backward:
with served_forward(model, served):
    loss = policy_gradient_loss(model(ids).logits, ...)
    loss.backward()
optimizer.step()
```

`sr_cast_bf16` adds 16 uniformly random bits to the fp32 bit pattern and truncates; the upper bf16 neighbour is chosen with probability equal to the fractional position, so an update of size Δ moves the served weight by Δ in expectation at any learning rate. `ema_update_` gives the explicit-lag variant of §7.5 (export an EMA of the master with the same unbiased rounding).

## Minimal end-to-end example

`examples/train_gsm8k.py` is a ~60-line GRPO loop on GSM8K (Qwen3-0.6B) that shows exactly where the two changes go. It uses Hugging Face `generate()` as the sampler, so it is slow; the paper's runs used vLLM.

```bash
pip install -e ".[example]"
python examples/train_gsm8k.py --mode nearest    # current framework default (round-to-nearest export)
python examples/train_gsm8k.py --mode faircast   # stochastic rounding + train on served copy
python examples/train_gsm8k.py --mode ef         # error feedback + train on served copy
```

A GPU smoke test of the package and the example runs on [Modal](https://modal.com) with `modal run scripts/modal_smoke.py` (`FAIRCAST_GPU` selects the GPU, default H100).

## Reproducing the paper's runs

`harness/` is the training harness used for every run in the paper, as run on Modal (one H100 per arm: vLLM sampler with sleep mode and a Hugging Face trainer on the same device).

* `harness/drift_rl.py`: GRPO loop, task rewards (Reasoning Gym, MBPP+/HumanEval+, MATH, GSM8K, synthetic games), export modes and the per-token logging behind Figure 4. Export is selected by `sync_mode` (`cast` = round-to-nearest, `sr` = stochastic rounding, `ef` = error feedback, `efsr` = dithered error feedback) and the trainer forward by `forward_mode` (`autocast` = bf16 autocast of the fp32 master, `ste` = served copy with straight-through gradients). Fair Cast is `sync_mode: "sr", forward_mode: "ste"`; `ema_export: 0.05` adds the explicit lag of §7.5.
* `harness/configs/`: the exact per-arm JSON configurations that produced every curve and table entry. Arm naming: `*_cast*` round-to-nearest baseline, `*_srste*` / `*_sr_*` Fair Cast, `*_efste*` error feedback, `*_fp32sampler` fp32-sampler oracle, `*_ema*` EMA export.
* `harness/eval_ckpt.py`: export-format evaluation of saved fp32 masters (Table 3). `harness/microbench_sync.py`: export-cost microbenchmark (§7.7). `harness/games_block.py`: the exact-solver game tasks.

```bash
pip install modal && modal setup
modal run harness/drift_rl.py --cfgs harness/configs/phase24_kk_q06.json   # knights-and-knaves, 0.6B, nearest vs Fair Cast
```

The harness expects two Modal volumes (`drift-loupe` for run outputs, `airbook-hf-cache` for the Hugging Face cache); rename them at the top of `drift_rl.py` for your workspace.

## Results and paper

* `results/TABLE2_ENDPOINTS.md`: endpoint accuracy of every matched nearest-vs-Fair-Cast seed pair.
* `paper/main.tex` + `paper/figures/`: the paper source. `make -C paper` builds `main.pdf` with pdflatex; `make -C paper arxiv` produces the flat `arxiv.tar.gz` for submission.
* `paper/scripts/`: the scripts that draw the figures. `gen_intro_figs.py` is self-contained; the others read the per-step training logs and held-out evaluations of the released runs (`results/evals.json`, `results/logs/`), which are distributed separately from this repository.

## Citation

```bibtex
@article{gao2026faircast,
  title   = {bf16 Rounding Silently Discards Most RL Policy Updates: A Hidden Training-Inference Mismatch and an Unbiased One-Line Fix},
  author  = {Gao, Andrew and Gao, William},
  year    = {2026},
  note    = {Preprint}
}
```

## License

MIT.
