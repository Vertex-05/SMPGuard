import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def safe_filename(name):
    """Convert a layer name into a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def main():
    parser = argparse.ArgumentParser(
        description="Plot BNP per-layer KL score distributions."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to kl_scores.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save plots. Defaults to a plots folder beside the CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else csv_path.parent / "plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_columns = {"layer", "kl_score"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"CSV must contain columns: {required_columns}"
        )

    df["kl_score"] = pd.to_numeric(
        df["kl_score"], errors="coerce"
    )
    df = df.dropna(subset=["layer", "kl_score"])

    if df.empty:
        raise ValueError("No valid KL scores found in CSV.")

    print(f"Loaded {len(df)} channel scores.")
    print(f"Number of layers: {df['layer'].nunique()}")
    print(f"Saving plots to: {out_dir}")

    # 1. Summary statistics for every layer
    summary = (
        df.groupby("layer")["kl_score"]
        .agg(
            channel_count="count",
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    summary.to_csv(
        out_dir / "kl_layer_summary.csv",
        index=False,
    )

    # 2. One histogram per layer
    for layer_name, group in df.groupby("layer", sort=False):
        scores = group["kl_score"].to_numpy()

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(scores, bins="auto", edgecolor="black")
        ax.axvline(
            scores.mean(),
            linestyle="--",
            label=f"Mean = {scores.mean():.4g}",
        )
        ax.axvline(
            pd.Series(scores).median(),
            linestyle=":",
            label=f"Median = {pd.Series(scores).median():.4g}",
        )

        ax.set_title(f"KL Score Distribution: {layer_name}")
        ax.set_xlabel("KL score")
        ax.set_ylabel("Number of channels")
        ax.legend()
        fig.tight_layout()

        filename = safe_filename(layer_name) + "_hist.png"
        fig.savefig(out_dir / filename, dpi=200)
        plt.close(fig)

    # 3. Boxplot comparing KL distributions across layers
    layers = list(df.groupby("layer", sort=False))
    layer_names = [name for name, _ in layers]
    layer_scores = [
        group["kl_score"].to_numpy()
        for _, group in layers
    ]

    fig_width = max(12, len(layer_names) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    ax.boxplot(
        layer_scores,
        labels=layer_names,
        showfliers=True,
    )

    ax.set_title("BNP KL Score Distribution Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("KL score")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()

    fig.savefig(
        out_dir / "kl_boxplot_all_layers.png",
        dpi=200,
    )
    plt.close(fig)

    # 4. Overall histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["kl_score"], bins="auto", edgecolor="black")
    ax.set_title("Overall BNP KL Score Distribution")
    ax.set_xlabel("KL score")
    ax.set_ylabel("Number of channels")
    fig.tight_layout()

    fig.savefig(
        out_dir / "kl_hist_all_channels.png",
        dpi=200,
    )
    plt.close(fig)

    print("Finished plotting.")
    print(f"Layer summary: {out_dir / 'kl_layer_summary.csv'}")
    print(f"All-layer boxplot: {out_dir / 'kl_boxplot_all_layers.png'}")
    print(f"Overall histogram: {out_dir / 'kl_hist_all_channels.png'}")


if __name__ == "__main__":
    main()