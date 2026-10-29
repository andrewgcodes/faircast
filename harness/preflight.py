"""Preflight: assert launch invariants for a phase cfg before gating. Usage: python3 preflight.py phaseX.json [harness]"""
import json, sys, ast, os
cfgs = json.load(open(sys.argv[1])); harness = sys.argv[2] if len(sys.argv) > 2 else "drift_rl.py"
ast.parse(open(harness).read()); ok = True
BIG = ("Qwen3-4B", "Qwen2.5-3B", "OLMoE", "MoE-A2.7B")
for c in cfgs:
    name = c["name"]; model = c.get("model", "Qwen/Qwen3-1.7B")
    if c.get("max_new", 512) < 2048: print(f"[{name}] max_new={c.get('max_new',512)} < 2048"); ok = False
    if any(b in model for b in BIG) and os.environ.get("DRIFT_GPU") != "H200": print(f"[{name}] big model {model} but DRIFT_GPU != H200"); ok = False
    if not c.get("save_final"): print(f"[{name}] save_final missing"); ok = False
    if c.get("sync_mode", "cast") != "cast" and c.get("forward_mode", "autocast") not in ("ste", "ste_swap") and not c.get("allow_no_ste"): print(f"[{name}] {c.get('sync_mode')} export without forward_mode=ste (half the method; collapses) — set forward_mode ste or allow_no_ste true"); ok = False
    if c.get("lora_r") and harness != "drift_lora.py": print(f"[{name}] lora cfg but harness {harness}"); ok = False
    if c.get("moe") and c.get("optimizer") != "adamw8bit": print(f"[{name}] moe without adamw8bit"); ok = False
    if "[train_prompts[(i) % len(train_prompts)]" not in open(harness).read(): print("harness lacks prompt wrap-around"); ok = False
print("PREFLIGHT", "OK" if ok else "FAILED", sys.argv[1]); sys.exit(0 if ok else 1)
