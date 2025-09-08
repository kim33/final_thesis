import matplotlib.pyplot as plt
import pandas as pd
import json

RAW_PATH = "dataset/categorize/paradetox_holistic_bias_summary.json"
REFUSED_PATH = "false_refusal_categorize/qwen/qwen_7B/qwen_7B_holistic_bias_paradetox_summary.json"
OUTPUT = "bias_analysis/qwen/qwen_7B_paradetox.csv"
MODEL="Qwen_7B"
DATASET="Paradetox"
IMAGE_OUTPUT=f"bias_analysis/qwen/{MODEL}_{DATASET}.png"

with open(RAW_PATH, "r") as f:
    raw_data = json.load(f)
raw_counts = raw_data[0].get("category_counts")

with open(REFUSED_PATH, "r") as f:
    refused_data = json.load(f)
refused_counts = refused_data[0].get("category_counts")

def analyze_false_refusals(raw_counts: dict, false_refusals: dict, save_path: str = None, save_plot: str = None):
    df = pd.DataFrame({
        "raw_count": pd.Series(raw_counts),
        "false_refusals": pd.Series(false_refusals)
    })

    # Totals
    N_raw = df["raw_count"].sum()
    N_fr = df["false_refusals"].sum()

    # Metrics
    df["refusal_rate"] = (df["false_refusals"] / df["raw_count"]).round(3)
    df["p_raw"] = (df["raw_count"] / N_raw).round(3)
    df["p_fr"] = (df["false_refusals"] / N_fr).round(3)
    df["bias_ratio"] = (df["p_fr"] / df["p_raw"]).round(3)

    # Plot
    plt.figure(figsize=(10,6))
    df["bias_ratio"].sort_values(ascending=False).plot(kind="bar", color="skyblue", edgecolor="black")
    plt.axhline(1, color="red", linestyle="--", linewidth=1)
    plt.title(f"{MODEL} {DATASET} False Refusals Bias Ratio by Category", fontsize=14)
    plt.ylabel("Bias Ratio (p_fr / p_raw)")
    plt.xlabel("Category")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_plot:
        plt.savefig(save_plot, dpi=300, bbox_inches="tight")

    plt.show()
    plt.show()

    # Save results if path given
    if save_path:
        if save_path.endswith(".csv"):
            df.to_csv(save_path, index=True)
        elif save_path.endswith(".xlsx"):
            df.to_excel(save_path, index=True)
        else:
            raise ValueError("Save path must end with .csv or .xlsx")

    return df

# Example usage (saves results as CSV)
_ = analyze_false_refusals(raw_counts, refused_counts, save_path=OUTPUT, save_plot=IMAGE_OUTPUT)
