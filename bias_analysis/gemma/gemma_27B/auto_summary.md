# Auto Summary

**Systematic over-representation (mean bias ratio):**
- religion: mean 1.78, max 2.00
- socioeconomic class: mean 1.64, max 2.27
- nationality: mean 1.48, max 2.00

**Consistent under-representation:**
- gender and sex: mean 0.77
- race ethnicity: mean 0.87

**Most stable models (lower SD of log-bias across datasets):**
- gemma3: SD=0.201

**Top overall fairness (lower is better):**
- gemma3: fairness_score=0.166

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
