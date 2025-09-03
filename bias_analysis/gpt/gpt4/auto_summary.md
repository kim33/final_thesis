# Auto Summary

**Systematic over-representation (mean bias ratio):**
- nationality: mean 1.67, max 3.00
- religion: mean 1.46, max 1.71
- sexual orientation: mean 1.43, max 2.00

**Consistent under-representation:**
- gender and sex: mean 0.76
- body type: mean 0.89

**Most stable models (lower SD of log-bias across datasets):**
- gpt4: SD=0.158

**Top overall fairness (lower is better):**
- gpt4: fairness_score=0.123

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
