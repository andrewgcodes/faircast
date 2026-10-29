# Table 2: the 25 matched nearest-vs-Fair-Cast seed pairs (Table 2 and §7.5 of the paper)

Scope: complete nearest-vs-Fair-Cast pairs on Qwen3-0.6B/1.7B/4B for the tasks shown in the paper. Excluded from this scope (runs outside the paper): Qwen3-8B zebra, Qwen2.5-1.5B MATH, OLMoE MATH, the 500-step Countdown pair, partial pairs. MBPP+ endpoints are on the 224 problems absent from the MBPP training split (121 of the first 128 for seed 0); other rows are the final served-copy eval as logged (per-arm curves in evals.json).

| task | model | seed | optimizer | n_eval | nearest final | Fair Cast final | Δ (pts) |
|---|---|---|---|---|---|---|---|
| games:nim | Qwen3-1.7B | 0 | adamw | 512 | 0.570 | 0.580 | +1.0 |
| gsm8k | Qwen3-0.6B | 0 | adamw | 512 | 0.742 | 0.742 | +0.0 |
| gsm8k | Qwen3-0.6B | 1 | adamw | 512 | 0.697 | 0.762 | +6.4 |
| gsm8k | Qwen3-0.6B | 2 | adamw | 512 | 0.699 | 0.789 | +9.0 |
| humanevalplus | Qwen3-1.7B | 0 | adamw | 164 | 0.585 | 0.616 | +3.0 |
| humanevalplus | Qwen3-1.7B | 1 | adamw | 164 | 0.640 | 0.634 | -0.6 |
| math | Qwen3-0.6B | 0 | adamw | 512 | 0.521 | 0.541 | +2.0 |
| math | Qwen3-0.6B | 1 | adamw | 512 | 0.521 | 0.543 | +2.1 |
| math | Qwen3-0.6B | 2 | adamw | 512 | 0.477 | 0.516 | +3.9 |
| math | Qwen3-1.7B | 0 | adamw | 128 | 0.664 | 0.656 | -0.8 |
| math_l5 | Qwen3-1.7B | 0 | adamw | 128 | 0.492 | 0.523 | +3.1 |
| mbppplus | Qwen3-1.7B | 0 | adamw | 128 | 0.587 | 0.612 | +2.5 |
| mbppplus | Qwen3-1.7B | 1 | adamw | 512 | 0.603 | 0.598 | -0.5 |
| mbppplus | Qwen3-1.7B | 2 | adamw | 512 | 0.598 | 0.629 | +3.1 |
| mbppplus | Qwen3-1.7B | 3 | adamw | 512 | 0.594 | 0.634 | +4.0 |
| rg:knights_knaves | Qwen3-0.6B | 0 | adamw | 512 | 0.865 | 0.967 | +10.2 |
| rg:knights_knaves | Qwen3-1.7B | 0 | adamw | 512 | 0.959 | 0.996 | +3.7 |
| rg:knights_knaves | Qwen3-1.7B | 0 | muon | 512 | 0.895 | 0.986 | +9.2 |
| rg:knights_knaves | Qwen3-1.7B | 1 | muon | 512 | 0.918 | 0.990 | +7.2 |
| rg:knights_knaves | Qwen3-4B | 0 | adamw | 512 | 0.988 | 0.992 | +0.4 |
| rg:list_functions | Qwen3-1.7B | 0 | adamw | 512 | 0.668 | 0.523 | -14.5 |
| rg:string_manipulation | Qwen3-1.7B | 0 | adamw | 512 | 0.352 | 0.363 | +1.2 |
| rg:zebra_puzzles | Qwen3-1.7B | 0 | adamw | 512 | 0.268 | 0.623 | +35.5 |
| rg:zebra_puzzles | Qwen3-1.7B | 1 | adamw | 512 | 0.447 | 0.625 | +17.8 |
| rg:zebra_puzzles | Qwen3-1.7B | 2 | adamw | 512 | 0.268 | 0.703 | +43.6 |

Tally: 25 pairs; ahead by more than one point 18; within one point 6; behind 1; mean Δ +6.1; median +3.1.
