from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"

OUTPUT_PATH = Path(
    "results/factconsolidation_inventory.json"
)


def get_source(row: dict[str, Any]) -> str:
    """Find the dataset/subtask name across possible schemas."""
    metadata = row.get("metadata") or {}

    candidates = [
        row.get("source"),
        row.get("dataset"),
        metadata.get("source"),
        metadata.get("dataset"),
        metadata.get("dataset_name"),
        metadata.get("subtask"),
    ]

    for candidate in candidates:
        if candidate is not None and str(candidate).strip():
            return str(candidate)

    return "UNKNOWN"


def preview(value: Any, length: int = 180) -> str:
    """Create a short one-line preview."""
    text = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )

    text = text.replace("\n", " ")

    if len(text) > length:
        return text[:length] + "..."

    return text


def main() -> None:
    print(
        f"Loading {DATASET_NAME}, "
        f"split={SPLIT_NAME}..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    print()
    print(f"Number of rows: {len(dataset)}")
    print(f"Columns: {dataset.column_names}")

    inventory: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()

    for index, dataset_row in enumerate(dataset):
        row = dict(dataset_row)

        context = row.get("context") or ""
        questions = row.get("questions") or []
        answers = row.get("answers") or []
        metadata = row.get("metadata") or {}

        source = get_source(row)
        source_counts[source] += 1

        first_question = (
            questions[0]
            if isinstance(questions, list) and questions
            else None
        )

        first_answer = (
            answers[0]
            if isinstance(answers, list) and answers
            else None
        )

        item = {
            "row_index": index,
            "source": source,
            "context_characters": len(context),
            "context_words": len(context.split()),
            "question_count": (
                len(questions)
                if isinstance(questions, list)
                else 0
            ),
            "answer_count": (
                len(answers)
                if isinstance(answers, list)
                else 0
            ),
            "metadata_keys": (
                sorted(metadata.keys())
                if isinstance(metadata, dict)
                else []
            ),
            "first_question": first_question,
            "first_answer": first_answer,
        }

        inventory.append(item)

        print()
        print("=" * 72)
        print(f"ROW INDEX: {index}")
        print(f"SOURCE: {source}")
        print(
            "CONTEXT CHARACTERS:",
            f"{len(context):,}",
        )
        print(
            "APPROXIMATE WORDS:",
            f"{len(context.split()):,}",
        )
        print(
            "NUMBER OF QUESTIONS:",
            item["question_count"],
        )
        print(
            "METADATA KEYS:",
            item["metadata_keys"],
        )
        print(
            "FIRST QUESTION:",
            preview(first_question),
        )
        print(
            "FIRST ANSWER:",
            preview(first_answer),
        )

    output = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "row_count": len(dataset),
        "columns": dataset.column_names,
        "source_counts": dict(source_counts),
        "rows": inventory,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("SOURCE COUNTS")

    for source, count in source_counts.items():
        print(f"{source}: {count}")

    print()
    print(f"Saved inventory to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()