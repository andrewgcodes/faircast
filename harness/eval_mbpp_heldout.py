"""Re-score saved MBPP+ served copies on the MBPP+ problems whose task_id is NOT in the MBPP train+validation+prompt union the runs trained on
(154/378 overlap by task_id; 224 held out). Same prompt format, greedy decoding, scorer and max_new as drift_rl.py; reports all/held-out/overlap accuracy.
Weights: /vol/runs/<run>/final_bf16.pt (step 150 served copy) else ckpt.pt['last_bf16'] (served copy at the last checkpoint, step 149).
Usage (from modal/evalck): modal run eval_mbpp_heldout.py --runs a,b,c"""
import modal, json, os
app = modal.App("drift-probe")
img = (modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
       .pip_install("torch==2.6.0", "vllm==0.8.5", "transformers==4.51.3", "datasets", "numpy"))
vol = modal.Volume.from_name("drift-loupe"); hf = modal.Volume.from_name("drift-hf", create_if_missing=True)
SYSTEM = "Solve the problem. Reason briefly, then give the final answer as \\boxed{...}."

def reward_mbpp_plus(text, gold, timeout=20.0):
    import re as _re, subprocess, tempfile, os as _os
    m = _re.findall(r"```(?:python)?\n(.*?)```", text, _re.S); code = m[-1] if m else text
    prog = (gold.get("setup") or "") + "\n" + code + "\n\n" + gold["test_code"] + "\nprint('__PASS__')\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    try:
        r = subprocess.run(["python3", path], capture_output=True, text=True, timeout=timeout, cwd=tempfile.gettempdir())
        return float("__PASS__" in r.stdout)
    except Exception:
        return 0.0
    finally:
        try: _os.unlink(path)
        except Exception: pass

@app.function(image=img, gpu="H100", timeout=3600, max_containers=3, volumes={"/vol": vol, "/hf": hf},
              env={"HF_HOME": "/hf", "VLLM_USE_V1": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_LOGGING_LEVEL": "WARNING"})
def eval_run(run: str):
    import torch, numpy as np
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from datasets import load_dataset
    cfg = json.load(open(f"/vol/runs/{run}/cfg.json")); model_id = cfg.get("model", "Qwen/Qwen3-1.7B"); max_new = cfg.get("max_new", 2048); n_eval = cfg.get("n_eval", 512)
    tok = AutoTokenizer.from_pretrained(model_id)
    def fmt(q):
        return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ds = load_dataset("google-research-datasets/mbpp", "full", split="train+validation+prompt"); ev = load_dataset("evalplus/mbppplus", split="test")
    train_ids = set(ds["task_id"])
    items = []
    for r in list(ev)[:n_eval]:
        q = (r["prompt"].strip() + "\n\nYour code should satisfy these tests:\n" + "\n".join(eval(r["test_list"]) if isinstance(r["test_list"], str) else r["test_list"]) + "\n\nWrite the full Python function in a single ```python code block.")
        gold = dict(setup="\n".join(eval(r["test_imports"]) if isinstance(r["test_imports"], str) else r["test_imports"]), test_code=r["test"])
        p = fmt(q)
        if len(tok(p).input_ids) <= 1024 - 16: items.append((p, gold, int(r["task_id"]) in train_ids))
    llm = LLM(model_id, gpu_memory_utilization=0.85, max_model_len=1024 + max_new, dtype="bfloat16", enable_prefix_caching=False, seed=0)
    if os.path.exists(f"/vol/runs/{run}/final_bf16.pt"):
        sd = torch.load(f"/vol/runs/{run}/final_bf16.pt", map_location="cpu"); tag = "final_bf16 (step 150)"
    else:
        sd = torch.load(f"/vol/runs/{run}/ckpt.pt", map_location="cpu")["last_bf16"]; tag = "ckpt last_bf16 (step 149)"
    mr = llm.llm_engine.model_executor.driver_worker.model_runner; mr.model.load_weights([(k, v) for k, v in sd.items()])
    res = llm.generate([p for p, _, _ in items], SamplingParams(temperature=0.0, max_tokens=max_new))
    hits = [reward_mbpp_plus(r.outputs[0].text, g) for r, (_, g, _) in zip(res, items)]
    ov = [o for _, _, o in items]
    out = dict(run=run, tag=tag, n_all=len(items), acc_all=float(np.mean(hits)),
               n_heldout=int(sum(not o for o in ov)), acc_heldout=float(np.mean([h for h, o in zip(hits, ov) if not o])),
               n_overlap=int(sum(ov)), acc_overlap=float(np.mean([h for h, o in zip(hits, ov) if o])), hits=[int(h) for h in hits], overlap=[int(o) for o in ov])
    with open(f"/vol/runs/{run}/eval_mbpp_heldout.json", "w") as f: json.dump(out, f)
    vol.commit(); return out

@app.local_entrypoint()
def main(runs: str):
    for r in eval_run.map(runs.split(","), return_exceptions=True):
        print(json.dumps({k: v for k, v in r.items() if k not in ("hits", "overlap")}) if isinstance(r, dict) else str(r))
