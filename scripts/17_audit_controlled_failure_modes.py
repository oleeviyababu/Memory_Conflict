from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CHAIN_PATH = Path(
    "results/14_update_chains.jsonl"
)

CONTROLLED_PATH = Path(
    "results/16_controlled_conflict_distractors.jsonl"
)

EARLIER_ISOLATED_PATH = Path(
    "results/10_gold_assisted_isolated.jsonl"
)

OUTPUT_PATH = Path(
    "results/17_controlled_failure_modes_summary.json"
)


CONDITION_ORDER = [
    "latest_only",
    "latest_plus_3_unrelated",
    "old_plus_latest",
    "old_plus_latest_plus_2_unrelated",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
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


def normalize(text: str) -> str:
    text = text.casefold().strip()

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return re.sub(r"\s+", " ", text).strip()


def contains_value(
    prediction: str,
    values: list[str],
) -> bool:
    normalized_prediction = normalize(prediction)

    return any(
        normalize(value) in normalized_prediction
        for value in values
        if normalize(value)
    )


def classify_answer(
    prediction: str,
    gold_answers: list[str],
    stale_values: list[str],
) -> str:
    if contains_value(
        prediction,
        gold_answers,
    ):
        return "latest"

    if contains_value(
        prediction,
        stale_values,
    ):
        return "stale"

    normalized_prediction = normalize(prediction)

    if normalized_prediction.startswith("unknown"):
        return "unknown"

    return "other"


def main() -> None:
    for path in [
        CHAIN_PATH,
        CONTROLLED_PATH,
        EARLIER_ISOLATED_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing file: {path}"
            )

    chains = {
        int(row["question_index"]): row
        for row in load_jsonl(CHAIN_PATH)
    }

    controlled_rows = load_jsonl(
        CONTROLLED_PATH
    )

    counts: dict[str, Counter[str]] = {
        condition: Counter()
        for condition in CONDITION_ORDER
    }

    examples: dict[
        str,
        dict[str, list[dict[str, Any]]],
    ] = {
        condition: {
            "stale": [],
            "unknown": [],
            "other": [],
        }
        for condition in CONDITION_ORDER
    }

    for result in controlled_rows:
        if result.get("group") != "conflict":
            continue

        condition = str(result["condition"])

        if condition not in counts:
            continue

        question_index = int(
            result["question_index"]
        )

        chain = chains[question_index]

        gold_answers = [
            str(value)
            for value in result["gold_answers"]
        ]

        stale_values = [
            str(statement["value"])
            for statement in chain.get(
                "previous_conflicting_statements",
                [],
            )
        ]

        prediction = str(
            result["model_answer"]
        )

        label = classify_answer(
            prediction=prediction,
            gold_answers=gold_answers,
            stale_values=stale_values,
        )

        counts[condition][label] += 1

        if (
            label in examples[condition]
            and len(examples[condition][label]) < 5
        ):
            examples[condition][label].append(
                {
                    "question_index":
                        question_index,
                    "question":
                        result["question"],
                    "gold_answers":
                        gold_answers,
                    "stale_values":
                        stale_values,
                    "model_answer":
                        prediction,
                }
            )

    earlier_isolated = {
        int(row["question_index"]): row
        for row in load_jsonl(
            EARLIER_ISOLATED_PATH
        )
        if not row.get("skipped", False)
    }

    current_latest = {
        int(row["question_index"]): row
        for row in controlled_rows
        if row["condition"] == "latest_only"
    }

    template_comparison = Counter()

    for question_index in sorted(
        set(earlier_isolated)
        & set(current_latest)
    ):
        earlier_correct = bool(
            earlier_isolated[question_index][
                "substring_exact_match"
            ]
        )

        current_correct = bool(
            current_latest[question_index][
                "substring_exact_match"
            ]
        )

        if earlier_correct and current_correct:
            category = "both_correct"
        elif earlier_correct:
            category = "earlier_prompt_only_correct"
        elif current_correct:
            category = "current_prompt_only_correct"
        else:
            category = "both_wrong"

        template_comparison[category] += 1

    summary = {
        "conflict_question_count": 74,
        "answer_classification": {
            condition: {
                "latest": counts[condition]["latest"],
                "stale": counts[condition]["stale"],
                "unknown": counts[condition]["unknown"],
                "other": counts[condition]["other"],
            }
            for condition in CONDITION_ORDER
        },
        "examples": examples,
        "isolated_prompt_template_comparison": {
            "earlier_prompt_accuracy":
                sum(
                    int(
                        row["substring_exact_match"]
                    )
                    for row in earlier_isolated.values()
                ),
            "current_prompt_accuracy":
                sum(
                    int(
                        row["substring_exact_match"]
                    )
                    for row in current_latest.values()
                ),
            "categories":
                dict(template_comparison),
            "warning": (
                "The evidence statement is derived from "
                "the same gold-assisted update chain, but "
                "the two prompt templates differ."
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 86)
    print("CONTROLLED ANSWER TYPES — CONFLICT QUESTIONS")
    print("=" * 86)
    print(
        f"{'Condition':<39}"
        f"{'Latest':>9}"
        f"{'Stale':>9}"
        f"{'UNKNOWN':>10}"
        f"{'Other':>9}"
    )

    for condition in CONDITION_ORDER:
        values = counts[condition]

        print(
            f"{condition:<39}"
            f"{values['latest']:>9}"
            f"{values['stale']:>9}"
            f"{values['unknown']:>10}"
            f"{values['other']:>9}"
        )

    print()
    print("=" * 86)
    print("ISOLATED PROMPT-TEMPLATE COMPARISON")
    print("=" * 86)

    for category, count in (
        template_comparison.items()
    ):
        print(f"{category}: {count}")

    print(f"\nSummary: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()