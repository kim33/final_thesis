# Auto Summary

**Systematic over-representation (mean bias ratio):**
- cultural: mean 1.58, max 2.20
- religion: mean 1.54, max 2.00
- political ideologies: mean 1.45, max 2.11

**Consistent under-representation:**
- sexual orientation: mean 0.70
- gender and sex: mean 0.73

**Most stable models (lower SD of log-bias across datasets):**
- mistral: SD=0.188

**Top overall fairness (lower is better):**
- mistral: fairness_score=0.096

**Next steps:**
- Target data augmentation and red-teaming for the top 3 over-represented categories.
- Calibrate refusal policies/thresholds and re-run this dashboard to measure movement.
- Inspect per-dataset slopegraphs where spikes appear.
