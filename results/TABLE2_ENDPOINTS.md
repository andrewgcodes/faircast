# Table 2 endpoints: every exact-matched nearest-vs-Fair-Cast seed pair (generated from the held-out evaluations of the released runs)

Fair = coin-flip rounding + train-on-served (SR+STE). Peak = best held-out checkpoint. Δ in accuracy points. Partial = run killed/stopped before step 149 (last common step shown).

| task | model | seed | opt | n_eval | nearest final (peak) | fair final (peak) | Δ final | Δ peak | status | other arms in cell |
|---|---|---|---|---|---|---|---|---|---|---|
| games:kuhn | Qwen3-1.7B | 0 | adamw | 512 | 0.555 (0.830@19) | 0.578 (0.795@9) | +2.3 | -3.5 | partial (nearest@39, fair@19) |  |
| games:nim | Qwen3-1.7B | 0 | adamw | 512 | 0.570 (0.596@119) | 0.580 (0.736@119) | +1.0 | +14.1 | complete |  |
| games:subtraction | Qwen3-1.7B | 0 | adamw | 512 | 0.371 (0.375@0) | 0.338 (0.451@79) | -3.3 | +7.6 | partial (nearest@99, fair@129) |  |
| gsm8k | Qwen3-0.6B | 0 | adamw | 512 | 0.742 (0.742@149) | 0.742 (0.768@99) | +0.0 | +2.5 | complete |  |
| gsm8k | Qwen3-0.6B | 1 | adamw | 512 | 0.697 (0.723@129) | 0.762 (0.781@119) | +6.4 | +5.9 | complete |  |
| gsm8k | Qwen3-0.6B | 2 | adamw | 512 | 0.699 (0.711@119) | 0.789 (0.789@150) | +9.0 | +7.8 | complete |  |
| humanevalplus | Qwen3-1.7B | 0 | adamw | 164 | 0.585 (0.665@49) | 0.616 (0.652@9) | +3.0 | -1.2 | complete | p19_heplus_efste_s0 |
| humanevalplus | Qwen3-1.7B | 1 | adamw | 164 | 0.640 (0.671@129) | 0.634 (0.659@39) | -0.6 | -1.2 | complete |  |
| math | Qwen2.5-1.5B-Instruct | 0 | adamw | 128 | 0.422 (0.422@149) | 0.414 (0.430@49) | -0.8 | +0.8 | complete |  |
| math | Qwen3-0.6B | 0 | adamw | 512 | 0.521 (0.527@119) | 0.541 (0.549@139) | +2.0 | +2.1 | complete |  |
| math | Qwen3-0.6B | 1 | adamw | 512 | 0.521 (0.551@139) | 0.543 (0.543@150) | +2.1 | -0.8 | complete |  |
| math | Qwen3-0.6B | 2 | adamw | 512 | 0.477 (0.500@139) | 0.516 (0.535@149) | +3.9 | +3.5 | complete |  |
| math | Qwen3-1.7B | 0 | adamw | 128 | 0.664 (0.703@124) | 0.656 (0.766@124) | -0.8 | +6.2 | complete | p10_cast_tis_lag2, p10_cast_tis_lag4, p10_srste_tis_lag4, p10_srste_tis_lag2, p11_math_srste_500, p11_math_cast_500, p11_math_efste_500 |
| math | OLMoE-1B-7B-0125-Instruct | 0 | adamw | 128 | 0.172 (0.227@49) | 0.242 (0.266@74) | +7.0 | +3.9 | complete |  |
| math_l5 | Qwen3-1.7B | 0 | adamw | 128 | 0.492 (0.531@99) | 0.523 (0.539@99) | +3.1 | +0.8 | complete |  |
| mbppplus | Qwen3-1.7B | 0 | adamw | 128 | 0.578 (0.602@24) | 0.617 (0.648@124) | +3.9 | +4.7 | complete |  |
| mbppplus | Qwen3-1.7B | 1 | adamw | 512 | 0.611 (0.611@149) | 0.603 (0.630@139) | -0.8 | +1.9 | complete |  |
| mbppplus | Qwen3-1.7B | 2 | adamw | 512 | 0.593 (0.611@99) | 0.643 (0.643@150) | +5.0 | +3.2 | complete |  |
| mbppplus | Qwen3-1.7B | 3 | adamw | 512 | 0.601 (0.614@109) | 0.643 (0.643@150) | +4.2 | +2.9 | complete |  |
| rg:countdown | Qwen3-1.7B | 0 | adamw | 128 | 0.227 (0.492@349) | 0.391 (0.406@499) | +16.4 | -8.6 | complete | p11_cd_efste_500 |
| rg:knights_knaves | Qwen3-0.6B | 0 | adamw | 512 | 0.865 (0.865@149) | 0.967 (0.967@150) | +10.2 | +10.2 | complete |  |
| rg:knights_knaves | Qwen3-1.7B | 0 | adamw | 512 | 0.959 (0.973@139) | 0.996 (0.996@139) | +3.7 | +2.3 | complete | p22_kk_fp32sampler |
| rg:knights_knaves | Qwen3-1.7B | 0 | muon | 512 | 0.895 (0.895@149) | 0.986 (0.988@149) | +9.2 | +9.4 | complete |  |
| rg:knights_knaves | Qwen3-1.7B | 1 | muon | 512 | 0.918 (0.918@149) | 0.990 (0.990@150) | +7.2 | +7.2 | complete |  |
| rg:knights_knaves | Qwen3-4B | 0 | adamw | 512 | 0.988 (0.990@59) | 0.992 (0.998@89) | +0.4 | +0.8 | complete |  |
| rg:list_functions | Qwen3-1.7B | 0 | adamw | 512 | 0.668 (0.678@139) | 0.523 (0.527@139) | -14.5 | -15.0 | complete | p26_lf_sr_ema05, p26_lf_sr_ema01 |
| rg:string_manipulation | Qwen3-1.7B | 0 | adamw | 512 | 0.352 (0.352@149) | 0.363 (0.363@149) | +1.2 | +1.2 | complete |  |
| rg:zebra_puzzles | Qwen3-1.7B | 0 | adamw | 512 | 0.268 (0.564@9) | 0.623 (0.645@129) | +35.5 | +8.0 | complete | p16_zebra_cast_tpen, p23_zebra_efste_s0, p23_zebra_fp32sampler, p25_zebra_srste_s0, p27_zebra_efsr_s0 |
| rg:zebra_puzzles | Qwen3-1.7B | 1 | adamw | 512 | 0.447 (0.562@9) | 0.625 (0.631@129) | +17.8 | +6.8 | complete | p25_zebra_efste_s1 |
| rg:zebra_puzzles | Qwen3-1.7B | 2 | adamw | 512 | 0.268 (0.580@19) | 0.703 (0.711@129) | +43.6 | +13.1 | complete |  |
| rg:zebra_puzzles | Qwen3-8B | 0 | adamw8bit | 512 | 0.637 (0.686@99) | 0.562 (0.695@39) | -7.4 | +1.0 | complete | p30c_zebra_q8b_srste_ema05 |

31 pairs (29 complete). Complete pairs: fair ahead on 20, tie (|Δ|≤1) 7, behind 2; mean Δ final +5.9.
