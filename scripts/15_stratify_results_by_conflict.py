from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


CHAIN_PATH = Path(
    "results/14_update_chains.jsonl"
)

CONDITION_PATHS = {
    "question_only": Path(
        "results/06_question_only_sh6k_all.jsonl"
    ),
    "full_context": Path(
        "results/05_full_context_sh6k_all.jsonl"
    ),
    "bm25_64": Path(
        "results/11_bm25_sh6k_budget64.jsonl"
    ),
    "bm25_128": Path(
        "results/11_bm25_sh6k_budget128.jsonl"
    ),
    "bm25_256": Path(
        "results/11_bm25_sh6k_budget256.jsonl"
    ),
    "bm25_512": Path(
        "results/08_bm25_sh6k_budget512.jsonl"
    ),
    "isolated_gold": Path(
        "results/10_gold_assisted_isolated.jsonl"
    ),
}

OUTPUT_JSON = Path(
    "results/15_conflict_stratification_summary.json"
)

OUTPUT_CSV = Path(
    "results/15_conflict_stratification_summary.csv"
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
    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

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


def row_is_correct(row: dict[str, Any]) -> bool:
    if row.get("skipped", False):
        return False

    if "substring_exact_match" not in row:
        raise KeyError(
            "Result row has no substring_exact_match field."
        )

    return bool(row["substring_exact_match"])


def summarize_group(
    indices: list[int],
    results: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    available_indices = [
        index
        for index in indices
        if index in results
        and not results[index].get("skipped", False)
    ]

    correct_count = sum(
        int(row_is_correct(results[index]))
        for index in available_indices
    )

    total = len(available_indices)

    return {
        "question_count": total,
        "correct_count": correct_count,
        "accuracy_percent": (
            100 * correct_count / total
            if total
            else None
        ),
    }


def paired_change(
    indices: list[int],
    lower: dict[int, dict[str, Any]],
    higher: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    both_correct = 0
    lower_only_correct = 0
    higher_only_correct = 0
    both_wrong = 0

    examples_lost: list[dict[str, Any]] = []
    examples_gained: list[dict[str, Any]] = []

    for index in indices:
        if index not in lower or index not in higher:
            continue

        lower_correct = row_is_correct(
            lower[index]
        )

        higher_correct = row_is_correct(
            higher[index]
        )

        if lower_correct and higher_correct:
            both_correct += 1

        elif lower_correct and not higher_correct:
            lower_only_correct += 1

            if len(examples_lost) < 10:
                examples_lost.append(
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

            if len(examples_gained) < 10:
                examples_gained.append(
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

    total = (
        both_correct
        + lower_only_correct
        + higher_only_correct
        + both_wrong
    )

    return {
        "question_count": total,
        "both_correct": both_correct,
        "lower_only_correct": lower_only_correct,
        "higher_only_correct": higher_only_correct,
        "both_wrong": both_wrong,
        "net_loss_count": (
            lower_only_correct
            - higher_only_correct
        ),
        "examples_lost": examples_lost,
        "examples_gained": examples_gained,
    }


def format_percent(value: Any) -> str:
    if value is None:
        return "N/A"

    return f"{value:.1f}%"


def main() -> None:
    chains = load_by_index(CHAIN_PATH)

    if len(chains) != 100:
        raise RuntimeError(
            f"Expected 100 update chains, found "
            f"{len(chains)}"
        )

    conflict_indices = sorted(
        index
        for index, row in chains.items()
        if int(
            row.get(
                "previous_conflict_count",
                0,
            )
        ) > 0
    )

    no_conflict_indices = sorted(
        index
        for index, row in chains.items()
        if int(
            row.get(
                "previous_conflict_count",
                0,
            )
        ) == 0
    )

    print(
        f"Conflict questions: {len(conflict_indices)}"
    )
    print(
        "No-conflict questions:",
        len(no_conflict_indices),
    )

    all_results = {
        name: load_by_index(path)
        for name, path in CONDITION_PATHS.items()
    }

    condition_summaries: list[dict[str, Any]] = []

    for condition, results in all_results.items():
        conflict_summary = summarize_group(
            indices=conflict_indices,
            results=results,
        )

        no_conflict_summary = summarize_group(
            indices=no_conflict_indices,
            results=results,
        )

        condition_summaries.append(
            {
                "condition": condition,
                "conflict": conflict_summary,
                "no_conflict": no_conflict_summary,
                "accuracy_gap_no_conflict_minus_conflict":
                    (
                        no_conflict_summary[
                            "accuracy_percent"
                        ]
                        - conflict_summary[
                            "accuracy_percent"
                        ]
                        if (
                            no_conflict_summary[
                                "accuracy_percent"
                            ]
                            is not None
                            and conflict_summary[
                                "accuracy_percent"
                            ]
                            is not None
                        )
                        else None
                    ),
            }
        )

    paired_64_to_512 = {
        "conflict": paired_change(
            indices=conflict_indices,
            lower=all_results["bm25_64"],
            higher=all_results["bm25_512"],
        ),
        "no_conflict": paired_change(
            indices=no_conflict_indices,
            lower=all_results["bm25_64"],
            higher=all_results["bm25_512"],
        ),
    }

    summary = {
        "question_count": len(chains),
        "conflict_question_count":
            len(conflict_indices),
        "no_conflict_question_count":
            len(no_conflict_indices),
        "conditions": condition_summaries,
        "paired_bm25_64_to_512":
            paired_64_to_512,
        "interpretation_warning": (
            "The gold answer was used upstream to identify "
            "the update chains and oracle statements. This "
            "is a diagnostic stratification."
        ),
    }

    OUTPUT_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_fields = [
        "condition",
        "conflict_question_count",
        "conflict_correct_count",
        "conflict_accuracy_percent",
        "no_conflict_question_count",
        "no_conflict_correct_count",
        "no_conflict_accuracy_percent",
        "accuracy_gap_no_conflict_minus_conflict",
    ]

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for item in condition_summaries:
            writer.writerow(
                {
                    "condition":
                        item["condition"],
                    "conflict_question_count":
                        item["conflict"][
                            "question_count"
                        ],
                    "conflict_correct_count":
                        item["conflict"][
                            "correct_count"
                        ],
                    "conflict_accuracy_percent":
                        item["conflict"][
                            "accuracy_percent"
                        ],
                    "no_conflict_question_count":
                        item["no_conflict"][
                            "question_count"
                        ],
                    "no_conflict_correct_count":
                        item["no_conflict"][
                            "correct_count"
                        ],
                    "no_conflict_accuracy_percent":
                        item["no_conflict"][
                            "accuracy_percent"
                        ],
                    "accuracy_gap_no_conflict_minus_conflict":
                        item[
                            "accuracy_gap_no_conflict_minus_conflict"
                        ],
                }
            )

    print()
    print("=" * 92)
    print("ACCURACY STRATIFIED BY UPDATE CONFLICT")
    print("=" * 92)
    print(
        f"{'Condition':<18}"
        f"{'Conflict (n=74)':>20}"
        f"{'No conflict (n=26)':>22}"
        f"{'Gap':>12}"
    )

    for item in condition_summaries:
        conflict = item["conflict"]
        no_conflict = item["no_conflict"]

        print(
            f"{item['condition']:<18}"
            f"{conflict['correct_count']:>4}/"
            f"{conflict['question_count']:<4} "
            f"{format_percent(conflict['accuracy_percent']):>7}"
            f"{no_conflict['correct_count']:>8}/"
            f"{no_conflict['question_count']:<4} "
            f"{format_percent(no_conflict['accuracy_percent']):>7}"
            f"{format_percent(item['accuracy_gap_no_conflict_minus_conflict']):>12}"
        )

    print()
    print("=" * 92)
    print("PAIRED CHANGE: BM25 64 → 512")
    print("=" * 92)

    for group, values in paired_64_to_512.items():
        print(
            f"{group:<12} | "
            f"lost={values['lower_only_correct']:>2} | "
            f"gained={values['higher_only_correct']:>2} | "
            f"net loss={values['net_loss_count']:>2} | "
            f"both wrong={values['both_wrong']:>2}"
        )

    print()
    print(f"JSON summary: {OUTPUT_JSON}")
    print(f"CSV summary:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()