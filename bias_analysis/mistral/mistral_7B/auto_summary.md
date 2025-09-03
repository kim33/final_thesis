# Auto Summary

**Systematic over-representation (mean bias ratio):**
- nationality: mean 1.94, max 3.00
- religion: mean 1.81, max 2.08
- political ideologies: mean 1.42, max 1.85

**Consistent under-representation:**
- gender and sex: mean 0.79
- age: mean 0.92

**Most stable models (lower SD of log-bias across datasets):**
- mistral: SD=0.124

**Top overall fairness (lower is better):**
- mistral: fairness_score=0.112

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
