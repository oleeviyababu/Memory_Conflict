from __future__ import annotations

import csv
import json
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any


BUDGET_PATHS = {
    64: Path(
        "results/11_bm25_sh6k_budget64.jsonl"
    ),
    128: Path(
        "results/11_bm25_sh6k_budget128.jsonl"
    ),
    256: Path(
        "results/11_bm25_sh6k_budget256.jsonl"
    ),
    512: Path(
        "results/08_bm25_sh6k_budget512.jsonl"
    ),
}

OUTPUT_SUMMARY = Path(
    "results/13_bm25_budget_pairwise_summary.json"
)

OUTPUT_TRAJECTORIES = Path(
    "results/13_bm25_budget_trajectories.jsonl"
)

OUTPUT_PAIRWISE_CSV = Path(
    "results/13_bm25_budget_pairwise.csv"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path}, "
                    f"line {line_number}"
                ) from error

    return rows


def load_by_index(
    path: Path,
) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(path)

    indexed = {
        int(row["question_index"]): row
        for row in rows
    }

    if len(indexed) != len(rows):
        raise RuntimeError(
            f"Duplicate question indices in {path}"
        )

    return indexed


def exact_mcnemar_p_value(
    lower_only_correct: int,
    higher_only_correct: int,
) -> float:
    """
    Two-sided exact McNemar/binomial test.

    Only discordant pairs are used:
    - lower budget correct, higher budget wrong
    - lower budget wrong, higher budget correct
    """
    discordant = (
        lower_only_correct
        + higher_only_correct
    )

    if discordant == 0:
        return 1.0

    smaller = min(
        lower_only_correct,
        higher_only_correct,
    )

    lower_tail = sum(
        comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2 * lower_tail)


def compare_pair(
    lower_budget: int,
    higher_budget: int,
    results: dict[
        int,
        dict[int, dict[str, Any]],
    ],
) -> dict[str, Any]:
    lower = results[lower_budget]
    higher = results[higher_budget]

    common_indices = sorted(
        set(lower) & set(higher)
    )

    both_correct = 0
    lower_only_correct = 0
    higher_only_correct = 0
    both_wrong = 0

    lost_examples: list[dict[str, Any]] = []
    gained_examples: list[dict[str, Any]] = []

    for index in common_indices:
        lower_correct = bool(
            lower[index]["substring_exact_match"]
        )
        higher_correct = bool(
            higher[index]["substring_exact_match"]
        )

        if lower_correct and higher_correct:
            both_correct += 1

        elif lower_correct and not higher_correct:
            lower_only_correct += 1

            if len(lost_examples) < 10:
                lost_examples.append(
                    {
                        "question_index": index,
                        "question": lower[index][
                            "question"
                        ],
                        "gold_answers": lower[index][
                            "gold_answers"
                        ],
                        "lower_answer": lower[index][
                            "model_answer"
                        ],
                        "higher_answer": higher[index][
                            "model_answer"
                        ],
                    }
                )

        elif not lower_correct and higher_correct:
            higher_only_correct += 1

            if len(gained_examples) < 10:
                gained_examples.append(
                    {
                        "question_index": index,
                        "question": lower[index][
                            "question"
                        ],
                        "gold_answers": lower[index][
                            "gold_answers"
                        ],
                        "lower_answer": lower[index][
                            "model_answer"
                        ],
                        "higher_answer": higher[index][
                            "model_answer"
                        ],
                    }
                )

        else:
            both_wrong += 1

    total = len(common_indices)

    lower_correct_total = (
        both_correct + lower_only_correct
    )

    higher_correct_total = (
        both_correct + higher_only_correct
    )

    return {
        "lower_budget": lower_budget,
        "higher_budget": higher_budget,
        "question_count": total,
        "lower_correct": lower_correct_total,
        "higher_correct": higher_correct_total,
        "accuracy_difference_points": (
            100
            * (
                lower_correct_total
                - higher_correct_total
            )
            / total
        ),
        "both_correct": both_correct,
        "lower_only_correct":
            lower_only_correct,
        "higher_only_correct":
            higher_only_correct,
        "both_wrong": both_wrong,
        "discordant_count": (
            lower_only_correct
            + higher_only_correct
        ),
        "exact_mcnemar_p_value":
            exact_mcnemar_p_value(
                lower_only_correct,
                higher_only_correct,
            ),
        "examples_lost_when_budget_increased":
            lost_examples,
        "examples_gained_when_budget_increased":
            gained_examples,
    }


def main() -> None:
    results: dict[
        int,
        dict[int, dict[str, Any]],
    ] = {}

    for budget, path in BUDGET_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing result file: {path}"
            )

        results[budget] = load_by_index(path)

    shared_indices = set.intersection(
        *(
            set(rows)
            for rows in results.values()
        )
    )

    if len(shared_indices) != 100:
        raise RuntimeError(
            "Expected 100 shared questions, found "
            f"{len(shared_indices)}"
        )

    budgets = sorted(BUDGET_PATHS)

    trajectory_counts: Counter[str] = Counter()
    trajectory_rows: list[dict[str, Any]] = []

    for index in sorted(shared_indices):
        correctness = {
            budget: bool(
                results[budget][index][
                    "substring_exact_match"
                ]
            )
            for budget in budgets
        }

        trajectory = "".join(
            "1" if correctness[budget] else "0"
            for budget in budgets
        )

        trajectory_counts[trajectory] += 1

        row = {
            "question_index": index,
            "question":
                results[64][index]["question"],
            "gold_answers":
                results[64][index]["gold_answers"],
            "trajectory": trajectory,
            "budget_order": budgets,
            "correctness": correctness,
            "answers": {
                budget: results[budget][index][
                    "model_answer"
                ]
                for budget in budgets
            },
        }

        trajectory_rows.append(row)

    with OUTPUT_TRAJECTORIES.open(
        "w",
        encoding="utf-8",
    ) as file:
        for row in trajectory_rows:
            file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    pairs = [
        (64, 128),
        (128, 256),
        (256, 512),
        (64, 256),
        (64, 512),
        (128, 512),
    ]

    pairwise_results = [
        compare_pair(
            lower_budget=lower,
            higher_budget=higher,
            results=results,
        )
        for lower, higher in pairs
    ]

    with OUTPUT_PAIRWISE_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        fields = [
            "lower_budget",
            "higher_budget",
            "question_count",
            "lower_correct",
            "higher_correct",
            "accuracy_difference_points",
            "both_correct",
            "lower_only_correct",
            "higher_only_correct",
            "both_wrong",
            "discordant_count",
            "exact_mcnemar_p_value",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for comparison in pairwise_results:
            writer.writerow(
                {
                    field: comparison[field]
                    for field in fields
                }
            )

    summary = {
        "budget_order": budgets,
        "trajectory_encoding": (
            "One digit per budget: "
            "1=correct, 0=wrong"
        ),
        "trajectory_counts": dict(
            trajectory_counts.most_common()
        ),
        "pairwise_comparisons":
            pairwise_results,
        "statistical_warning": (
            "McNemar tests treat questions as paired units. "
            "All questions share one benchmark context, so "
            "these p-values are diagnostic rather than final "
            "population-level evidence."
        ),
        "trajectory_file":
            str(OUTPUT_TRAJECTORIES),
        "pairwise_csv":
            str(OUTPUT_PAIRWISE_CSV),
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("PAIRED BM25 BUDGET COMPARISON")
    print("=" * 92)

    for comparison in pairwise_results:
        print(
            f"{comparison['lower_budget']:>3} "
            f"vs "
            f"{comparison['higher_budget']:<3} | "
            f"lost="
            f"{comparison['lower_only_correct']:>2} | "
            f"gained="
            f"{comparison['higher_only_correct']:>2} | "
            f"difference="
            f"{comparison['accuracy_difference_points']:+.1f} "
            f"points | "
            f"p="
            f"{comparison['exact_mcnemar_p_value']:.4f}"
        )

    print()
    print("Most common correctness trajectories")
    print(
        "Budget order:",
        budgets,
    )

    for trajectory, count in (
        trajectory_counts.most_common(10)
    ):
        print(
            f"  {trajectory}: {count}"
        )

    print()
    print(
        "Example: trajectory 1100 means correct at "
        "64 and 128 tokens, then wrong at 256 and "
        "512 tokens."
    )

    print(f"\nSummary:      {OUTPUT_SUMMARY}")
    print(f"Trajectories: {OUTPUT_TRAJECTORIES}")
    print(f"Pairwise CSV: {OUTPUT_PAIRWISE_CSV}")


if __name__ == "__main__":
    main()