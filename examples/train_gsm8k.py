"""Minimal Fair Cast reference: GRPO on GSM8K with (1) an unbiased fp32->bf16 export and (2) the loss computed on the served copy.
Single GPU, Hugging Face generate() as the sampler (swap in vLLM by loading the served state dict into it after each sync).
Not a production trainer: it exists to show exactly where the two changes go. Run: python train_gsm8k.py --mode faircast|nearest"""
import argparse, re, torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from faircast import export_bf16, served_forward

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen3-0.6B"); p.add_argument("--mode", default="faircast", choices=["faircast", "nearest", "ef"])
p.add_argument("--steps", type=int, default=150); p.add_argument("--prompts", type=int, default=8); p.add_argument("--group", type=int, default=8)
p.add_argument("--lr", type=float, default=1e-6); p.add_argument("--max_new", type=int, default=1024); p.add_argument("--seed", type=int, default=0)
a = p.parse_args(); torch.manual_seed(a.seed); dev = "cuda"
tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
master = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(dev)          # fp32 master (optimizer state lives here)
served = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(dev)         # bf16 served copy (sampler AND trainer forward)
opt = torch.optim.AdamW(master.parameters(), lr=a.lr, betas=(0.9, 0.999), weight_decay=0.01)
gen = torch.Generator(device=dev).manual_seed(a.seed)
resid = {n: torch.zeros_like(w) for n, w in master.named_parameters()} if a.mode == "ef" else None

@torch.no_grad()
def export():
    """The one-line change. nearest: served <- bf16(master). faircast: stochastic rounding, E[served] = master. ef: error feedback."""
    for (n, w), (_, q) in zip(master.named_parameters(), served.named_parameters()):
        q.copy_(export_bf16(w.data, {"nearest": "nearest", "faircast": "sr", "ef": "ef"}[a.mode], gen, None if resid is None else resid[n]))

ds = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
def answer(s):
    m = re.findall(r"\\boxed\{([^{}]*)\}", s); return m[-1].replace(",", "").strip() if m else None
def gold(s): return s.split("####")[-1].replace(",", "").strip()
def chat(q): return tok.apply_chat_template([{"role": "user", "content": q + "\n\nGive the final answer inside \\boxed{}."}], tokenize=False, add_generation_prompt=True)

export()
for step in range(a.steps):
    batch = ds.select(range(step * a.prompts, (step + 1) * a.prompts))
    prompts = [chat(q) for q in batch["question"]]; golds = [gold(x) for x in batch["answer"]]
    enc = tok(prompts, return_tensors="pt", padding=True).to(dev)
    with torch.no_grad():                                                                         # rollouts from the SERVED copy
        out = served.generate(**enc, do_sample=True, temperature=1.0, max_new_tokens=a.max_new, num_return_sequences=a.group, pad_token_id=tok.pad_token_id)
    plen = enc.input_ids.shape[1]; resp = out[:, plen:]
    rewards = torch.tensor([float(answer(tok.decode(r, skip_special_tokens=True)) == golds[i // a.group]) for i, r in enumerate(resp)], device=dev)
    adv = (rewards.view(-1, a.group) - rewards.view(-1, a.group).mean(1, keepdim=True)); adv = (adv / (adv.std(1, keepdim=True) + 1e-4)).view(-1)
    mask = (resp != tok.pad_token_id).float()
    # Loss on the SERVED copy with a straight-through estimator: the forward uses exactly the bf16 weights that generated the data
    # and the gradient lands on the fp32 master (w_eff = master + (served - master).detach()). The master is restored on exit.
    opt.zero_grad()
    with served_forward(master, {n: q for n, q in served.named_parameters()}):
        logits = master(out, attention_mask=(out != tok.pad_token_id).long()).logits[:, plen - 1:-1].float()
        logp = torch.gather(F.log_softmax(logits, -1), 2, resp.unsqueeze(-1)).squeeze(-1)
        loss = -((logp * adv.unsqueeze(1)) * mask).sum() / mask.sum()                              # one on-policy step (ratio == 1), GRPO advantages
        loss.backward()
    torch.nn.utils.clip_grad_norm_(master.parameters(), 1.0); opt.step()
    export()                                                                                       # re-export after every optimizer step
    print(f"step {step:3d}  reward {rewards.mean():.3f}  len {mask.sum(1).mean():.0f}", flush=True)
