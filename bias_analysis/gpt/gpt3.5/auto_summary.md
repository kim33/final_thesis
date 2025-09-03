# Auto Summary

**Systematic over-representation (mean bias ratio):**
- nationality: mean 1.83, max 3.00
- religion: mean 1.61, max 2.33
- sexual orientation: mean 1.36, max 2.08

**Consistent under-representation:**
- gender and sex: mean 0.79
- body type: mean 0.91

**Most stable models (lower SD of log-bias across datasets):**
- gpt3.5: SD=0.118

**Top overall fairness (lower is better):**
- gpt3.5: fairness_score=0.083

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
