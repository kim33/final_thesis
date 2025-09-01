
import json
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Paths
data_path = Path("false_refusal_categorize/llama/phi-4/llama3_70B_detox_refusal_category.json")
outdir = "false_refusal_categorize/llama/phi-4/diagram"

with open(data_path, "r") as f:
    data = json.load(f)

# Normalize to DataFrames
rows_cat = []
rows_bucket = []
rows_pair = []

for rec in data:
    key = (rec.get("model"), rec.get("task"), rec.get("dataset"))
    model, task, dataset = key
    # category_counts
    for cat, cnt in rec.get("category_counts", {}).items():
        rows_cat.append({"model": model, "task": task, "dataset": dataset, "category": cat, "count": cnt})
    # bucket_counts
    for b, cnt in rec.get("bucket_counts", {}).items():
        rows_bucket.append({"model": model, "task": task, "dataset": dataset, "bucket": b, "count": cnt})
    # category_bucket_pairs (nested)
    for cat, buckets in rec.get("category_bucket_pairs", {}).items():
        for b, cnt in buckets.items():
            rows_pair.append({"model": model, "task": task, "dataset": dataset, "category": cat, "bucket": b, "count": cnt})

df_cat = pd.DataFrame(rows_cat)
df_bucket = pd.DataFrame(rows_bucket)
df_pair = pd.DataFrame(rows_pair)


# Remove rows whose category or bucket is in the exclude set
print("Category counts by model/task/dataset")
print(df_cat.head())

print("\nBucket counts by model/task/dataset")
print(df_bucket.head())

print("\nCategory–Bucket pair counts")
print(df_pair.head())

# 1) Heatmap of categories vs dataset key
df_cat["key"] = df_cat[["model","task","dataset"]].agg("-".join, axis=1)
pivot = df_cat.pivot_table(index="category", columns="key", values="count", fill_value=0, aggfunc="sum")

fig1 = plt.figure(figsize=(10, max(4, 0.4*len(pivot))))  # height scales with number of categories
ax1 = plt.gca()
im = ax1.imshow(pivot.values, aspect="auto")
ax1.set_yticks(np.arange(len(pivot.index)))
ax1.set_yticklabels(pivot.index)
ax1.set_xticks(np.arange(len(pivot.columns)))
ax1.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax1.set_title("Category counts heatmap")
plt.colorbar(im, ax=ax1)
heatmap_path = outdir+"/category_heatmap.png"
plt.tight_layout()
plt.savefig(heatmap_path, dpi=150)
plt.show()

# 2) Horizontal bar of total category counts across all datasets
totals = df_cat.groupby("category")["count"].sum().sort_values()
fig2 = plt.figure(figsize=(8, max(4, 0.4*len(totals))))
ax2 = plt.gca()
ax2.barh(totals.index, totals.values)
ax2.set_title("Total category counts (all datasets)")
ax2.set_xlabel("Count")
bar_totals_path = outdir+"/total_category_counts.png"
plt.tight_layout()
plt.savefig(bar_totals_path, dpi=150)
plt.show()

#plot_pair_heatmap_overall
pivot = df_pair.pivot_table(index="category", columns="bucket", values="count",
                                aggfunc="sum", fill_value=0).sort_index()
fig = plt.figure(figsize=(max(10, 0.45*len(pivot.columns)),
                              max(6, 0.4*len(pivot.index))))
ax = plt.gca()
im = ax.imshow(pivot.values, aspect="auto")
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_yticklabels(pivot.index)
ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
ax.set_title("Category × Bucket counts (overall)")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(outdir+"/pair_heatmap_overall.png", dpi=150)
plt.close(fig)

#plot_pair_heatmap_per_key
for key, sub in df_pair.groupby(["model","task","dataset"]):
        key_str = "-".join(key)
        pivot = sub.pivot_table(index="category", columns="bucket", values="count",
                                aggfunc="sum", fill_value=0).sort_index()
        if pivot.empty:
            continue
        fig = plt.figure(figsize=(max(10, 0.45*len(pivot.columns)),
                                  max(6, 0.4*len(pivot.index))))
        ax = plt.gca()
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
        ax.set_title(f"Category × Bucket — {key_str}")
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(outdir+ f"/pair_heatmap_{key_str}.png", dpi=150)
        plt.close(fig)


#    For each category (overall), show a stacked horizontal bar of top-N buckets.
#    Remaining buckets are grouped into 'other'.
top_buckets_per_cat = 10

cats = sorted(df_pair["category"].unique())
    # Build per-category top-N table
stacks = []
bucket_labels = set()
for cat in cats:
        sub = df_pair[df_pair["category"] == cat].groupby("bucket")["count"].sum().sort_values(ascending=False)
        top = sub.head(top_buckets_per_cat)
        other = sub.iloc[top_buckets_per_cat:].sum()
        row = top.to_dict()
        if other > 0:
            row["other"] = other
        stacks.append(row)
        bucket_labels.update(row.keys())

bucket_labels = list(sorted(bucket_labels))  # consistent order
    # Build matrix

M = np.zeros((len(cats), len(bucket_labels)), dtype=float)
for i, row in enumerate(stacks):
        for j, b in enumerate(bucket_labels):
            M[i, j] = row.get(b, 0)

fig = plt.figure(figsize=(12, max(6, 0.45*len(cats))))
ax = plt.gca()
left = np.zeros(len(cats))
bars_handles = []
for j, b in enumerate(bucket_labels):
        h = ax.barh(cats, M[:, j], left=left, label=b)
        left += M[:, j]
        bars_handles.append(h)

ax.set_title(f"Bucket composition per category (top {top_buckets_per_cat} buckets + other)")
ax.set_xlabel("Count")
ax.legend(loc="best", ncol=2, fontsize=8)
plt.tight_layout()
plt.savefig(outdir+ f"/category_bucket_composition_top{top_buckets_per_cat}.png", dpi=150)
plt.close(fig)

# 3) For each dataset key: horizontal bar of top 12 buckets
plots_saved = []
for key, sub in df_bucket.groupby(["model","task","dataset"]):
    key_str = "-".join(key)
    topN = sub.sort_values("count", ascending=False).head(12)
    fig = plt.figure(figsize=(9, max(4, 0.35*len(topN))))
    ax = plt.gca()
    bars = ax.barh(topN["bucket"][::-1], topN["count"][::-1])  # reverse for largest at top
    ax.set_title(f"Top 12 buckets — {key_str}")
    ax.set_xlabel("Count")
    ax.bar_label(bars, padding=2) 
    plt.tight_layout()
    out_path = outdir+f"/top_buckets_{key_str}.png".replace(" ", "_")
    plt.savefig(out_path, dpi=150)
    plots_saved.append(out_path)
    plt.show()

# 4) Also provide a grouped bar of categories per dataset key for quick comparison
# (Not stacked; separate bars per category within each dataset key would be too dense; instead, plot per key)
plots_saved2 = []
for key, sub in df_cat.groupby(["model","task","dataset"]):
    key_str = "-".join(key)
    ordered = sub.sort_values("count", ascending=False)
    fig = plt.figure(figsize=(10, 4 + 0.15*len(ordered)))
    ax = plt.gca()
    bars = ax.barh(ordered["category"][::-1], ordered["count"][::-1])
    ax.set_title(f"Category counts — {key_str}")
    ax.set_xlabel("Count")
    ax.bar_label(bars, padding=2)
    plt.tight_layout()
    out_path = outdir+f"/category_counts_{key_str}.png".replace(" ", "_")
    plt.savefig(out_path, dpi=150)
    plots_saved2.append(out_path)
    plt.show()


def plot_bucket_bars_per_category(df_pair: pd.DataFrame, topn: int = 10):
    """
    For each category (overall), plot a horizontal bar chart of its top-N buckets.
    """
    if df_pair.empty:
        return

    for cat, sub in df_pair.groupby("category"):
        top = (sub.groupby("bucket")["count"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(topn))
        fig, ax = plt.subplots(figsize=(9, 0.5*len(top) + 2))
        bars = ax.barh(top.index[::-1], top.values[::-1])
        ax.set_title(f"Top {topn} buckets for category: {cat}")
        ax.set_xlabel("Count")
        # annotate
        try:
            ax.bar_label(bars, padding=2)
        except Exception:
            for b in bars:
                w = b.get_width()
                ax.text(w, b.get_y()+b.get_height()/2, f"{int(w)}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir+f"/pair_bar_category_{cat}.png", dpi=150)
        plt.close(fig)

def plot_category_bars_per_bucket(df_pair: pd.DataFrame, topn: int = 8):
    """
    For each bucket (overall), plot a horizontal bar chart of its top-N categories.
    """
    if df_pair.empty:
        return

    for buc, sub in df_pair.groupby("bucket"):
        top = (sub.groupby("category")["count"]
                 .sum()
                 .sort_values(ascending=False)
                 .head(topn))
        fig, ax = plt.subplots(figsize=(9, 0.5*len(top) + 2))
        bars = ax.barh(top.index[::-1], top.values[::-1])
        ax.set_title(f"Top {topn} categories for bucket: {buc}")
        ax.set_xlabel("Count")
        # annotate
        try:
            ax.bar_label(bars, padding=2)
        except Exception:
            for b in bars:
                w = b.get_width()
                ax.text(w, b.get_y()+b.get_height()/2, f"{int(w)}", va="center", ha="left", fontsize=9)
        plt.tight_layout()
        plt.savefig(outdir+f"/pair_bar_bucket_{buc}.png", dpi=150)
        plt.close(fig)


def plot_grouped_bars_per_category_across_keys(df_pair: pd.DataFrame, top_buckets: int = 10):
    """
    For each category, build a grouped bar chart:
      x-axis = keys (model-task-dataset),
      groups = keys,
      bars in each group = top buckets for that category (overall top across keys).
    """
    if df_pair.empty:
        return

    # Ensure a 'key' column
    if "key" not in df_pair.columns:
        df_pair["key"] = df_pair[["model","task","dataset"]].agg("-".join, axis=1)

    for cat, sub in df_pair.groupby("category"):
        # Choose top buckets for this category across all keys
        top_b = (sub.groupby("bucket")["count"].sum()
                   .sort_values(ascending=False).head(top_buckets).index.tolist())
        sub = sub[sub["bucket"].isin(top_b)]

        # Pivot: rows=key, cols=bucket, values=count
        pivot = sub.pivot_table(index="key", columns="bucket", values="count",
                                aggfunc="sum", fill_value=0).reindex(columns=top_b)

        if pivot.empty or len(pivot.index) == 0:
            continue

        # Plot grouped bars
        keys = pivot.index.tolist()
        buckets = pivot.columns.tolist()
        x = np.arange(len(keys))
        width = max(0.1, min(0.8 / max(1, len(buckets)), 0.18))  # bar width fits groups

        fig, ax = plt.subplots(figsize=(max(10, 0.8*len(keys)), 6))
        for i, b in enumerate(buckets):
            vals = pivot[b].values
            bars = ax.bar(x + (i - (len(buckets)-1)/2)*width, vals, width=width, label=b)
            # annotate on top
            try:
                ax.bar_label(bars, padding=2)
            except Exception:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, h, f"{int(h)}",
                                ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(keys, rotation=45, ha="right")
        ax.set_title(f"Category '{cat}': top {len(buckets)} buckets across keys")
        ax.set_ylabel("Count")
        ax.legend(loc="best", fontsize=8, ncol=min(3, len(buckets)))
        plt.tight_layout()
        plt.savefig(outdir+f"/pair_grouped_category_{cat}.png", dpi=150)
        plt.close(fig)

if not df_pair.empty and "key" not in df_pair.columns:
    df_pair["key"] = df_pair[["model","task","dataset"]].agg("-".join, axis=1)

# Category↔Bucket bar charts
plot_bucket_bars_per_category(df_pair, topn=10)      # Option A
#plot_category_bars_per_bucket(df_pair, topn=10)      # Option B
plot_grouped_bars_per_category_across_keys(df_pair, top_buckets=10)
# Summarize created files to show download links in the chat message
files_created = [heatmap_path, bar_totals_path] + plots_saved + plots_saved2
files_created
