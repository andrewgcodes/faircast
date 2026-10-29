"""GPU smoke test on Modal: kernel self-test + unit tests + a few GRPO steps of examples/train_gsm8k.py in all three export modes.
Usage: modal run scripts/modal_smoke.py [--gpu H100] [--steps 2]"""
import os, subprocess, modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = modal.App("faircast-smoke")
img = (modal.Image.debian_slim(python_version="3.11")
       .pip_install("torch==2.6.0", "transformers==4.51.3", "datasets", "accelerate", "pytest")
       .add_local_dir(os.path.join(ROOT, "faircast"), remote_path="/root/faircast")
       .add_local_dir(os.path.join(ROOT, "examples"), remote_path="/root/examples")
       .add_local_dir(os.path.join(ROOT, "tests"), remote_path="/root/tests"))


@app.function(image=img, gpu=os.environ.get("FAIRCAST_GPU", "H100"), timeout=3600, env={"PYTHONPATH": "/root", "HF_HOME": "/root/hf"})
def smoke(steps: int = 2) -> dict:
    import torch
    out = {"gpu": torch.cuda.get_device_name()}
    def sh(cmd):
        print(f"\n$ {cmd}", flush=True)
        r = subprocess.run(cmd, shell=True, cwd="/root", capture_output=True, text=True)
        print(r.stdout[-4000:], r.stderr[-3000:], flush=True)
        return r.returncode, r.stdout
    out["selftest"] = sh("python -m faircast")[0]
    out["pytest"] = sh("python -m pytest -q tests")[0]
    for mode in ("nearest", "faircast", "ef"):
        rc, so = sh(f"python examples/train_gsm8k.py --mode {mode} --steps {steps} --prompts 2 --group 4 --max_new 128")
        out[f"train_{mode}"] = rc
        out[f"train_{mode}_tail"] = so.strip().splitlines()[-steps:] if so.strip() else []
    return out


@app.local_entrypoint()
def main(steps: int = 2):
    import json
    print(json.dumps(smoke.remote(steps), indent=1))
