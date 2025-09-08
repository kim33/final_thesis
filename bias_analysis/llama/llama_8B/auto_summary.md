# Auto Summary

**Systematic over-representation (mean bias ratio):**
- sexual orientation: mean 1.39, max 1.92
- socioeconomic class: mean 1.25, max 1.43
- race ethnicity: mean 1.24, max 1.36

**Consistent under-representation:**
- gender and sex: mean 0.79
- body type: mean 0.94

**Most stable models (lower SD of log-bias across datasets):**
- llama3: SD=0.127

**Top overall fairness (lower is better):**
- llama3: fairness_score=0.089

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
