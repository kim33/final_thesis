import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from scipy.stats import ks_2samp, mannwhitneyu

REFUSED_PATH = "qwen/qwen_30B_paradetox_ling.json"
ORIGINAL_PATH = "../../dataset/linguistic/paradetox_linguistic.json"
OUTPATH = "qwen/qwen_30B"
MODEL = "Qwen 30B"
DATASET = "Paradetox"

os.makedirs(OUTPATH, exist_ok=True)

def load_dataset(path, label):
    with open(path, "r") as f:
        data = json.load(f)
    # keep only proper records; skip any summary rows
    records = [
        {
            "dataset": label,  # <- lowercase and consistent
            "total_tokens": d["total_tokens"],
            "clause_count": d["clause_count"],
            "avg_parse_tree_depth": d["avg_parse_tree_depth"],
        }
        for d in data
        if isinstance(d, dict) and "total_tokens" in d
    ]
    return records

refused = load_dataset(REFUSED_PATH, "Refused")
original = load_dataset(ORIGINAL_PATH, "Original")  # <- use "Original" label
df = pd.DataFrame(refused + original)

# ensure numeric (guards against unexpected types)
for col in ["total_tokens", "clause_count", "avg_parse_tree_depth"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

sns.set(style="whitegrid", palette="Set2")
features = ["total_tokens", "clause_count", "avg_parse_tree_depth"]

# ---- PLOTS + SAVE ----
for feature in features:
    plt.figure(figsize=(8, 5))
    ax = sns.kdeplot(
        data=df,
        x=feature,
        hue="dataset",              # <- matches column name
        fill=True,
        common_norm=False,
        alpha=0.4,
        linewidth=1.5,
        legend=True
    )

    plt.title(f"{DATASET} — {feature.replace('_', ' ').title()} Distribution", fontsize=14, weight="bold")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Density")

    # model name annotation inside the plot
    plt.text(
        0.98, 0.98, MODEL,
        ha="right", va="top",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
        style="italic",
        weight="bold",
        alpha=0.9
    )

    plt.tight_layout()
    save_path = os.path.join(OUTPATH, f"{DATASET}_{feature}_distribution.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved: {save_path}")

# ---- STATS ----
print("\n== Statistical tests (Refused vs Original) ==")
for feature in features:
    x = df.loc[df["dataset"] == "Refused", feature].dropna()
    y = df.loc[df["dataset"] == "Original", feature].dropna()

    if x.empty or y.empty:
        print(f"{feature}: insufficient data (Refused n={len(x)}, Original n={len(y)})")
        continue

    ks_stat, ks_p = ks_2samp(x, y)
    u_stat, u_p = mannwhitneyu(x, y, alternative='two-sided')

    print(f"{feature}: KS p={ks_p:.4g} | Mann-Whitney p={u_p:.4g}  (n_refused={len(x)}, n_orig={len(y)})")
