# Auto Summary

**Systematic over-representation (mean bias ratio):**
- political ideologies: mean 1.48, max 2.31
- nationality: mean 1.37, max 2.11
- religion: mean 1.33, max 2.33

**Consistent under-representation:**
- gender and sex: mean 0.80
- characteristics: mean 0.91

**Most stable models (lower SD of log-bias across datasets):**
- qwen: SD=0.242

**Top overall fairness (lower is better):**
- qwen: fairness_score=0.187

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
