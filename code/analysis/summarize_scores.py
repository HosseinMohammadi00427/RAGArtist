"""Reproduce the six Figure 11 group means from the processed evaluation CSV."""

import argparse
import csv
from collections import defaultdict
from statistics import mean


METRICS = (
    "AttributeBinding",
    "Spatial",
    "NonSpatial",
    "Color_fuzzy",
    "Shape_fuzzy",
    "Texture_fuzzy",
)


def summarize(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))

    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[row["prompt"]].append(row)

    groups = {"ImageRAG base": [], "GenArtist last correction": [], "Selected final": []}
    for prompt_rows in by_prompt.values():
        base = next(row for row in prompt_rows if int(row["edit#"]) == 0)
        groups["ImageRAG base"].append(base)

        corrections = [row for row in prompt_rows if int(row["edit#"]) > 0]
        if corrections:
            groups["GenArtist last correction"].append(
                max(corrections, key=lambda row: int(row["edit#"]))
            )

        selected_index = int(base["best"])
        selected = next(
            row for row in prompt_rows if int(row["edit#"]) == selected_index
        )
        groups["Selected final"].append(selected)

    for label, group_rows in groups.items():
        scores = ", ".join(
            f"{metric}={mean(float(row[metric]) for row in group_rows):.4f}"
            for metric in METRICS
        )
        print(f"{label} (n={len(group_rows)}): {scores}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to evaluations_processed.csv")
    summarize(parser.parse_args().csv_path)
