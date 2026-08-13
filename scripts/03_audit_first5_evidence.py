from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

RESULTS_PATH = Path(
    "results/02_full_context_sh6k_smoke.jsonl"
)
OUTPUT_PATH = Path(
    "results/03_first5_evidence_audit.json"
)

METADATA_FIELDS = [
    "question_ids",
    "qa_pair_ids",
    "question_types",
    "question_dates",
    "keypoints",
    "previous_events",
]


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}

        if metadata.get("source") == SOURCE_NAME:
            return row

    raise RuntimeError(
        f"Could not find source {SOURCE_NAME}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                results.append(json.loads(line))

    return results


def preview(value: Any, limit: int = 800) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        default=str,
    )

    if len(text) <= limit:
        return text

    return text[:limit] + "..."


def get_aligned_metadata(
    metadata: dict[str, Any],
    field: str,
    question_index: int,
) -> Any:
    """
    Try to obtain the metadata element associated with one question.
    If its structure is not question-aligned, return a preview of
    the complete field instead of guessing.
    """
    value = metadata.get(field)

    if isinstance(value, list):
        if len(value) == 100:
            return value[question_index]

        return {
            "note": (
                "This list is not length 100, so it may not be "
                "directly aligned with questions."
            ),
            "length": len(value),
            "preview": value[:3],
        }

    if isinstance(value, dict):
        question_ids = metadata.get("question_ids")

        if (
            isinstance(question_ids, list)
            and len(question_ids) > question_index
        ):
            question_id = str(
                question_ids[question_index]
            )

            if question_id in value:
                return value[question_id]

        return {
            "note": "Dictionary structure",
            "keys_preview": list(value.keys())[:10],
        }

    return value


def find_snippets(
    text: str,
    search_term: str,
    radius: int = 300,
    maximum: int = 5,
) -> list[str]:
    """
    Find case-insensitive occurrences and return nearby context.
    """
    if not search_term.strip():
        return []

    lowered_text = text.casefold()
    lowered_term = search_term.casefold()

    snippets: list[str] = []
    start_position = 0

    while len(snippets) < maximum:
        position = lowered_text.find(
            lowered_term,
            start_position,
        )

        if position == -1:
            break

        snippet_start = max(0, position - radius)
        snippet_end = min(
            len(text),
            position + len(search_term) + radius,
        )

        snippet = text[
            snippet_start:snippet_end
        ].replace("\n", " ")

        snippets.append(snippet)

        start_position = position + len(search_term)

    return snippets


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing result file: {RESULTS_PATH}"
        )

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    context = str(row["context"])
    metadata = row.get("metadata") or {}
    results = load_jsonl(RESULTS_PATH)

    audits: list[dict[str, Any]] = []

    for result in results:
        question_index = int(
            result["question_index"]
        )
        question = result["question"]
        gold_answers = result["gold_answers"]
        model_answer = result["model_answer"]

        print()
        print("=" * 80)
        print(f"QUESTION {question_index + 1}")
        print("Question:", question)
        print("Gold:", gold_answers)
        print("Model:", model_answer)
        print(
            "Correct:",
            result["substring_exact_match"],
        )

        gold_searches: list[dict[str, Any]] = []

        for gold_answer in gold_answers:
            snippets = find_snippets(
                context,
                str(gold_answer),
            )

            gold_search = {
                "gold_answer": gold_answer,
                "occurrence_count_shown": len(
                    snippets
                ),
                "snippets": snippets,
            }

            gold_searches.append(gold_search)

            print()
            print(
                f"Gold answer '{gold_answer}' "
                f"found {len(snippets)} time(s), "
                "up to the display limit."
            )

            for snippet_index, snippet in enumerate(
                snippets,
                start=1,
            ):
                print(
                    f"\nGold snippet "
                    f"{snippet_index}:"
                )
                print(snippet)

        question_metadata: dict[str, Any] = {}

        print()
        print("QUESTION-SPECIFIC METADATA")

        for field in METADATA_FIELDS:
            value = get_aligned_metadata(
                metadata=metadata,
                field=field,
                question_index=question_index,
            )

            question_metadata[field] = value

            print()
            print(f"{field}:")
            print(preview(value))

        audits.append(
            {
                "question_index": question_index,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": model_answer,
                "correct": result[
                    "substring_exact_match"
                ],
                "gold_searches": gold_searches,
                "metadata": question_metadata,
            }
        )

    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "source": SOURCE_NAME,
                "context_characters": len(context),
                "audits": audits,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 80)
    print(f"Saved evidence audit to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()