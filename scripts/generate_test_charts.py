import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_boxplot(out_path: str) -> None:
    rng = np.random.default_rng(0)
    data = [
        rng.normal(loc=50, scale=10, size=120),
        rng.normal(loc=65, scale=12, size=120),
        rng.normal(loc=40, scale=8, size=120),
    ]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=200)
    bp = ax.boxplot(
        data,
        labels=["A", "B", "C"],
        patch_artist=True,
        showfliers=True,
        widths=0.5,
    )
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.6)
    for k in ("whiskers", "caps", "medians"):
        for line in bp[k]:
            line.set_color("#333333")
            line.set_linewidth(1.6)

    ax.set_title("Boxplot Demo", pad=12, fontsize=16)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_truncated_axis_bar(out_path: str) -> None:
    categories = ["A", "B", "C", "D"]
    values = np.array([100, 103, 107, 110], dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=200)
    ax.bar(categories, values, color="#4C78A8", edgecolor="#333333", linewidth=0.8)

    # Truncate: do not include 0 in the y-axis.
    ax.set_ylim(95, 112)
    ax.set_yticks([95, 100, 105, 110])
    ax.set_title("Truncated Axis Example", pad=12, fontsize=16)
    ax.set_xlabel("Category", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_inverted_axis_line(out_path: str) -> None:
    years = np.array([2018, 2019, 2020, 2021, 2022], dtype=int)
    values = np.array([10, 12, 13, 15, 17], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    ax.plot(years, values, marker="o", color="#4C78A8", linewidth=2.0)
    ax.set_title("Inverted Axis Example", pad=12, fontsize=16)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_xticks(years)
    ax.set_yticks([10, 12, 14, 16, 18])

    # Invert the y-axis (misleading).
    ax.invert_yaxis()
    ax.grid(axis="both", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_inconsistent_tick_intervals(out_path: str) -> None:
    x = np.array([1, 2, 3, 4, 5], dtype=int)
    y = np.array([15, 35, 55, 65, 85], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    ax.plot(x, y, marker="s", color="#54A24B", linewidth=2.0)
    ax.set_title("Inconsistent Tick Intervals Example", pad=12, fontsize=16)
    ax.set_xlabel("Index", fontsize=12)
    ax.set_ylabel("Value", fontsize=12)
    ax.set_xticks(x)

    # Misleading: equal spacing on the axis but inconsistent numeric intervals.
    ax.set_ylim(0, 100)
    tick_pos = [0, 20, 40, 60, 80, 100]
    tick_lbl = ["0", "10", "30", "40", "60", "100"]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lbl)

    ax.grid(axis="both", linestyle="--", alpha=0.25)
    ax.tick_params(axis="both", labelsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_dual_axis(out_path: str) -> None:
    years = np.array([2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022], dtype=int)
    london = np.array([600, 550, 650, 700, 680, 620, 580, 600], dtype=float)
    new_york = np.array([1000, 950, 900, 850, 800, 750, 700, 680], dtype=float)

    fig, ax1 = plt.subplots(figsize=(9.5, 6.5), dpi=200)
    ax2 = ax1.twinx()

    bars = ax2.bar(years, new_york, width=0.45, color="#72B7B2", alpha=0.55, edgecolor="#333333", linewidth=0.8)
    (line,) = ax1.plot(years, london, marker="o", color="#1B998B", linewidth=2.0)

    ax1.set_title("Annual rainfall in selected cities", pad=12, fontsize=16)
    ax1.set_xlabel("Years", fontsize=12)
    ax1.set_ylabel("Rainfall in millimeters (London)", fontsize=12, color="#1B998B")
    ax2.set_ylabel("Rainfall in millimeters (New York)", fontsize=12, color="#4E9E9B")

    ax1.set_xticks(years)
    ax1.tick_params(axis="y", labelcolor="#1B998B")
    ax2.tick_params(axis="y", labelcolor="#4E9E9B")
    ax1.grid(axis="both", linestyle="--", alpha=0.25)

    # Combined legend.
    ax1.legend([line, bars], ["London", "New York"], loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def make_radial_bar(out_path: str) -> None:
    categories = ["Europe", "Asia", "N. America", "S. America", "Africa", "Australia"]
    values = np.array([82, 55, 40, 68, 30, 48], dtype=float)
    max_val = 100.0

    # Matplotlib polar bar chart.
    N = len(categories)
    angles = np.linspace(0.0, 2.0 * np.pi, N, endpoint=False)
    width = (2.0 * np.pi / float(N)) * 0.85
    colors = ["#E45756", "#72B7B2", "#4C78A8", "#F58518", "#54A24B", "#B279A2"]

    fig = plt.figure(figsize=(9.5, 6.5), dpi=200)
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(angles, values, width=width, bottom=0.0, color=colors, alpha=0.95)

    ax.set_ylim(0, max_val)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=10)

    # Keep category labels horizontal-ish and readable for OCR.
    ax.set_xticks(angles)
    ax.set_xticklabels(categories, fontsize=11)

    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title("Radial Bar Demo", pad=18, fontsize=16)

    # Add legend with horizontal text (helps annotate_legend / OCR).
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.95) for c in colors]
    ax.legend(
        legend_handles,
        categories,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.join("image", "generated"))
    ap.add_argument("--manifest-out", default=os.path.join("out", "generated_smoke_manifest.jsonl"))
    args = ap.parse_args()

    out_dir = str(args.out_dir)
    _ensure_dir(out_dir)

    rows: List[Dict[str, Any]] = []

    def _add(name: str, fn, *, y_true: List[str], chart_type: List[str]) -> None:
        out_path = os.path.join(out_dir, name)
        fn(out_path)
        rows.append(
            {
                "image_path": out_path,
                "split": "dev",
                "y_true": y_true,
                "chart_type": chart_type,
            }
        )
        print(f"Wrote: {out_path}")

    # Normal charts (y_true empty) for sanity.
    _add("boxplot.png", make_boxplot, y_true=[], chart_type=["box plot"])
    _add("radial_bar.png", make_radial_bar, y_true=[], chart_type=["radial bar chart"])

    # Misleader smoke cases (minimal set from the plan).
    _add("truncated_axis_bar.png", make_truncated_axis_bar, y_true=["truncated axis"], chart_type=["bar chart"])
    _add("inverted_axis_line.png", make_inverted_axis_line, y_true=["inverted axis"], chart_type=["line chart"])
    _add(
        "inconsistent_tick_intervals.png",
        make_inconsistent_tick_intervals,
        y_true=["inconsistent tick intervals"],
        chart_type=["line chart"],
    )
    _add("dual_axis.png", make_dual_axis, y_true=["dual axis"], chart_type=["dual axis"])

    _write_jsonl(str(args.manifest_out), rows)
    print(f"Wrote manifest: {args.manifest_out}")


if __name__ == "__main__":
    main()
