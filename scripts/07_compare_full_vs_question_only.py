from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


FULL_PATH = Path(
    "results/05_full_context_sh6k_all.jsonl"
)
QUESTION_ONLY_PATH = Path(
    "results/06_question_only_sh6k_all.jsonl"
)

OUTPUT_PATH = Path(
    "results/07_full_vs_question_only.jsonl"
)
SUMMARY_PATH = Path(
    "results/07_full_vs_question_only_summary.json"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
        flags=re.UNICODE,
    )
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main() -> None:
    full_results = {
        int(row["question_index"]): row
        for row in load_jsonl(FULL_PATH)
    }

    question_only_results = {
        int(row["question_index"]): row
        for row in load_jsonl(QUESTION_ONLY_PATH)
    }

    indices = sorted(
        set(full_results)
        & set(question_only_results)
    )

    categories = {
        "both_correct": 0,
        "full_context_only_correct": 0,
        "question_only_only_correct": 0,
        "both_wrong": 0,
    }

    same_answer_count = 0
    comparison_rows: list[dict[str, Any]] = []

    for index in indices:
        full = full_results[index]
        question_only = question_only_results[index]

        full_correct = bool(
            full["substring_exact_match"]
        )

        question_only_correct = bool(
            question_only["substring_exact_match"]
        )

        if full_correct and question_only_correct:
            category = "both_correct"
        elif full_correct:
            category = "full_context_only_correct"
        elif question_only_correct:
            category = "question_only_only_correct"
        else:
            category = "both_wrong"

        categories[category] += 1

        same_answer = (
            normalize(full["model_answer"])
            == normalize(
                question_only["model_answer"]
            )
        )

        if same_answer:
            same_answer_count += 1

        row = {
            "question_index": index,
            "question": full["question"],
            "gold_answers": full["gold_answers"],
            "full_context_answer":
                full["model_answer"],
            "full_context_correct":
                full_correct,
            "question_only_answer":
                question_only["model_answer"],
            "question_only_correct":
                question_only_correct,
            "same_normalized_answer":
                same_answer,
            "category": category,
        }

        comparison_rows.append(row)

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for row in comparison_rows:
            output_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(comparison_rows)

    full_correct_count = sum(
        int(row["full_context_correct"])
        for row in comparison_rows
    )

    question_only_correct_count = sum(
        int(row["question_only_correct"])
        for row in comparison_rows
    )

    summary = {
        "question_count": total,
        "full_context_correct":
            full_correct_count,
        "full_context_accuracy_percent": (
            100 * full_correct_count / total
        ),
        "question_only_correct":
            question_only_correct_count,
        "question_only_accuracy_percent": (
            100 * question_only_correct_count
            / total
        ),
        "absolute_memory_gain_points": (
            100
            * (
                full_correct_count
                - question_only_correct_count
            )
            / total
        ),
        "same_answer_count": same_answer_count,
        "same_answer_percent": (
            100 * same_answer_count / total
        ),
        "categories": categories,
        "comparison_file": str(OUTPUT_PATH),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FULL CONTEXT VS QUESTION ONLY")
    print()
    print(
        f"Full context: "
        f"{full_correct_count}/{total} "
        f"({summary['full_context_accuracy_percent']:.1f}%)"
    )
    print(
        f"Question only: "
        f"{question_only_correct_count}/{total} "
        f"({summary['question_only_accuracy_percent']:.1f}%)"
    )
    print(
        "Net accuracy difference:",
        f"{summary['absolute_memory_gain_points']:+.1f}",
        "percentage points",
    )
    print()
    print("Outcome categories:")

    for category, count in categories.items():
        print(f"  {category}: {count}")

    print()
    print(
        "Same answer with and without memory:",
        f"{same_answer_count}/{total} "
        f"({summary['same_answer_percent']:.1f}%)",
    )

    print()
    print(f"Detailed comparison: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()