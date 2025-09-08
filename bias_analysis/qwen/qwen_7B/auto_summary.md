# Auto Summary

**Systematic over-representation (mean bias ratio):**
- nationality: mean 1.76, max 3.00
- sexual orientation: mean 1.33, max 1.92
- religion: mean 1.29, max 1.67

**Consistent under-representation:**
- gender and sex: mean 0.81
- body type: mean 0.93

**Most stable models (lower SD of log-bias across datasets):**
- qwen: SD=0.146

**Top overall fairness (lower is better):**
- qwen: fairness_score=0.074

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
