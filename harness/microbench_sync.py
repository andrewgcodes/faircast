"""Sync-cost microbenchmark on one GPU: nearest cast vs coin-flip (sr_round) vs carry-the-error (EF) over all parameters of Qwen3-1.7B, plus vLLM-free timing of the per-tensor work. Writes /vol/runs/microbench_sync.json."""
import modal, os, json
from drift_rl import app, img, vol, hf, sr_round
img = img.add_local_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_rl.py"), remote_path="/root/drift_rl.py", copy=True)
@app.function(image=img, gpu=os.environ.get("DRIFT_GPU", "H100"), timeout=1800, volumes={"/vol": vol, "/hf": hf}, env={"HF_HOME": "/hf"})
def bench(model_id: str = "Qwen/Qwen3-1.7B"):
    import torch, time
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).cuda()
    params = {n: p.data for n, p in m.named_parameters()}; n_params = sum(p.numel() for p in params.values())
    res = {n: torch.zeros_like(p) for n, p in params.items()}
    def nearest(): return [p.to(torch.bfloat16) for p in params.values()]
    def coinflip(): return [sr_round(p) for p in params.values()]
    def carry():
        out = []
        for n, p in params.items():
            r = res[n]; q = (p + r).to(torch.bfloat16); r.copy_(p + r - q.float()); out.append(q)
        return out
    gen = torch.Generator(device="cuda"); gen.manual_seed(0)
    def sr_bits(x):
        bits = x.contiguous().view(torch.int32)
        noise = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device, generator=gen)
        return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)
    def coinflip_bits(): return [sr_bits(p) for p in params.values()]
    def ef_sr():
        out = []
        for n, p in params.items():
            r = res[n]; y = p + r; q = sr_round(y); r.copy_(y - q.float()); out.append(q)
        return out
    def timeit(fn, k=10):
        fn(); torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(k): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / k * 1000
    out = dict(model=model_id, n_params=n_params, gpu=torch.cuda.get_device_name(), ms={k: timeit(f) for k, f in [("nearest", nearest), ("coinflip", coinflip), ("coinflip_bits", coinflip_bits), ("carry", carry), ("dithered_carry", ef_sr)]},
               extra_mem_bytes_carry=sum(r.numel() * 4 for r in res.values()))
    print(json.dumps(out)); 
    with open("/vol/runs/microbench_sync2.json", "w") as f: json.dump(out, f)
    vol.commit(); return out
@app.local_entrypoint()
def microbench(model_id: str = "Qwen/Qwen3-1.7B"): print(bench.remote(model_id))
