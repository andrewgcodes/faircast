"""Export-format eval of saved checkpoints: does the fp32 master's drift survive each export?
For each run: load runs/<name>/ckpt.pt (fp32 master at its last step) and evaluate greedy held-out accuracy of
  bf16   = nearest bf16 cast of master (what the sampler served)
  fp16   = fp16 cast of master (10 mantissa bits: keeps most of the sub-bf16 residual)
  fp8_bf = vLLM online FP8 from the bf16 cast      fp8_f16 = vLLM online FP8 from the fp16 cast
plus final_bf16.pt (the served copy actually saved) as the reference. Results -> /vol/runs/<name>/export_eval.json
Usage: modal run --detach eval_ckpt.py --names p19_heplus_cast_s0,p11_cd_cast_500"""
import modal, os, json, gc
from drift_rl import app, img, vol, hf, build_prompts, reward_fn
img = img.add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_rl.py"), remote_path="/root/drift_rl.py", copy=True)

@app.function(image=img, gpu=os.environ.get("DRIFT_GPU", "H100"), timeout=6*3600, volumes={"/vol": vol, "/hf": hf},
              env={"HF_HOME": "/hf", "VLLM_LOGGING_LEVEL": "WARNING", "TOKENIZERS_PARALLELISM": "false", "VLLM_USE_V1": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
def eval_exports(name: str):
    import torch, numpy as np
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM, SamplingParams
    out = f"/vol/runs/{name}"; cfg = json.load(open(f"{out}/cfg.json")); model_id = cfg.get("model", "Qwen/Qwen3-1.7B"); max_new = cfg.get("max_new", 2048)
    tok = AutoTokenizer.from_pretrained(model_id)
    _, eval_prompts = build_prompts(tok, 8, cfg.get("n_eval", 200), task=cfg.get("task", "gsm8k"))
    ck = torch.load(f"{out}/ckpt.pt", map_location="cpu", weights_only=False); master = ck["model"]; step = ck["step"]
    served = torch.load(f"{out}/final_bf16.pt", map_location="cpu") if os.path.exists(f"{out}/final_bf16.pt") else None
    sp = SamplingParams(temperature=0.0, max_tokens=max_new); res = dict(name=name, step=step, task=cfg.get("task"), n_eval=len(eval_prompts), variants={})
    def run(tag, state, dtype, quant=None):
        d = f"/tmp/exp_{tag}"
        m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        missing, unexpected = m.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
        m.to(dtype).save_pretrained(d, safe_serialization=True); tok.save_pretrained(d); del m; gc.collect()
        llm = LLM(d, dtype=str(dtype).split(".")[-1], quantization=quant, gpu_memory_utilization=0.5, max_model_len=1024 + max_new, enforce_eager=True, seed=0)
        outs = llm.generate([p for p, _ in eval_prompts], sp)
        hits = [reward_fn(r.outputs[0].text, g) for r, (_, g) in zip(outs, eval_prompts)]
        acc = float(np.mean(hits)); ml = float(np.mean([len(r.outputs[0].token_ids) for r in outs]))
        res["variants"][tag] = dict(acc=acc, mean_len=ml, hits=[int(h) for h in hits], missing=len(missing), unexpected=len(unexpected))
        print(f"[{name}] {tag}: acc={acc:.4f} mean_len={ml:.0f} missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        del llm, outs; gc.collect(); torch.cuda.empty_cache()
        with open(f"{out}/export_eval.json", "w") as f: json.dump(res, f); vol.commit()
    if served is not None: run("served_final_bf16", served, torch.bfloat16)
    run("bf16_of_master", master, torch.bfloat16)
    run("fp16_of_master", master, torch.float16)
    run("fp8_of_bf16", master, torch.bfloat16, quant="fp8")
    run("fp8_of_fp16", master, torch.float16, quant="fp8")
    return {k: v["acc"] for k, v in res["variants"].items()}

@app.local_entrypoint()
def exports(names: str):
    for name, r in zip(names.split(","), eval_exports.map(names.split(","))): print(name, json.dumps(r))
