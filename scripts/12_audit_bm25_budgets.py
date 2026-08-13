from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rank_bm25 import BM25Okapi


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

RESULT_PATHS = {
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

OUTPUT_JSON = Path(
    "results/12_bm25_budget_audit_summary.json"
)

OUTPUT_CSV = Path(
    "results/12_bm25_budget_audit_summary.csv"
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
                    f"Invalid JSON in {path} "
                    f"at line {line_number}"
                ) from error

    return rows


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}

        if metadata.get("source") == SOURCE_NAME:
            return row

    raise RuntimeError(
        f"Could not find source: {SOURCE_NAME}"
    )


def parse_numbered_statements(
    context: str,
) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?<!\d)(\d+)\.\s+"
        r"(.*?)"
        r"(?=\s+\d+\.\s+|\Z)",
        flags=re.DOTALL,
    )

    statements: list[dict[str, Any]] = []

    for match in pattern.finditer(context):
        statements.append(
            {
                "number": int(match.group(1)),
                "text": (
                    match.group(2)
                    .strip()
                    .replace("\n", " ")
                ),
            }
        )

    return statements


def lexical_tokens(text: str) -> list[str]:
    return re.findall(
        r"\w+",
        text.casefold(),
        flags=re.UNICODE,
    )


def normalize(text: str) -> str:
    text = text.casefold().strip()

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return re.sub(r"\s+", " ", text).strip()


def as_gold_list(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(item) for item in answer]

    return [str(answer)]


def contains_gold(
    text: str,
    gold_answers: list[str],
) -> bool:
    normalized_text = normalize(text)

    return any(
        normalize(gold) in normalized_text
        for gold in gold_answers
        if normalize(gold)
    )


def select_gold_assisted_statement(
    question: str,
    gold_answers: list[str],
    statements: list[dict[str, Any]],
    bm25: BM25Okapi,
) -> dict[str, Any] | None:
    """
    Diagnostic oracle.

    This uses the gold answer and must not be described as a
    deployable retrieval method.
    """
    scores = bm25.get_scores(
        lexical_tokens(question)
    )

    candidates: list[dict[str, Any]] = []

    for index, statement in enumerate(statements):
        if not contains_gold(
            statement["text"],
            gold_answers,
        ):
            continue

        candidates.append(
            {
                "number": statement["number"],
                "text": statement["text"],
                "score": float(scores[index]),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["score"],
            item["number"],
        ),
        reverse=True,
    )

    return candidates[0]


def audit_budget(
    budget: int,
    result_path: Path,
    questions: list[Any],
    answers: list[Any],
    statements: list[dict[str, Any]],
    bm25: BM25Okapi,
) -> dict[str, Any]:
    if not result_path.exists():
        raise FileNotFoundError(
            f"Missing result file: {result_path}"
        )

    results = {
        int(row["question_index"]): row
        for row in load_jsonl(result_path)
    }

    correct_count = 0
    oracle_resolved_count = 0
    oracle_present_count = 0
    oracle_present_correct = 0
    oracle_missing_count = 0
    oracle_missing_correct = 0

    statement_counts: list[int] = []
    evidence_token_counts: list[int] = []

    for question_index in range(len(questions)):
        result = results.get(question_index)

        if result is None:
            raise RuntimeError(
                f"Budget {budget} is missing "
                f"question {question_index}"
            )

        question = str(questions[question_index])
        gold_answers = as_gold_list(
            answers[question_index]
        )

        oracle = select_gold_assisted_statement(
            question=question,
            gold_answers=gold_answers,
            statements=statements,
            bm25=bm25,
        )

        retrieved_numbers = {
            int(item["number"])
            for item in result["retrieved_statements"]
        }

        is_correct = bool(
            result["substring_exact_match"]
        )

        if is_correct:
            correct_count += 1

        statement_counts.append(
            int(result["retrieved_statement_count"])
        )

        evidence_token_counts.append(
            int(result["evidence_token_count"])
        )

        if oracle is None:
            continue

        oracle_resolved_count += 1

        oracle_present = (
            int(oracle["number"])
            in retrieved_numbers
        )

        if oracle_present:
            oracle_present_count += 1

            if is_correct:
                oracle_present_correct += 1
        else:
            oracle_missing_count += 1

            if is_correct:
                oracle_missing_correct += 1

    total = len(questions)

    return {
        "budget": budget,
        "question_count": total,
        "correct_count": correct_count,
        "accuracy_percent": (
            100 * correct_count / total
        ),
        "oracle_resolved_count":
            oracle_resolved_count,
        "oracle_present_count":
            oracle_present_count,
        "oracle_recall_percent": (
            100
            * oracle_present_count
            / oracle_resolved_count
            if oracle_resolved_count
            else None
        ),
        "oracle_present_correct":
            oracle_present_correct,
        "accuracy_when_oracle_present_percent": (
            100
            * oracle_present_correct
            / oracle_present_count
            if oracle_present_count
            else None
        ),
        "oracle_missing_count":
            oracle_missing_count,
        "oracle_missing_correct":
            oracle_missing_correct,
        "accuracy_when_oracle_missing_percent": (
            100
            * oracle_missing_correct
            / oracle_missing_count
            if oracle_missing_count
            else None
        ),
        "minimum_statement_count":
            min(statement_counts),
        "maximum_statement_count":
            max(statement_counts),
        "mean_statement_count": (
            sum(statement_counts)
            / len(statement_counts)
        ),
        "minimum_evidence_tokens":
            min(evidence_token_counts),
        "maximum_evidence_tokens":
            max(evidence_token_counts),
        "mean_evidence_tokens": (
            sum(evidence_token_counts)
            / len(evidence_token_counts)
        ),
        "result_file": str(result_path),
    }


def format_optional(value: Any) -> str:
    if value is None:
        return "N/A"

    return f"{value:.1f}"


def main() -> None:
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    context = str(row["context"])
    questions = row["questions"]
    answers = row["answers"]

    statements = parse_numbered_statements(
        context
    )

    corpus = [
        lexical_tokens(statement["text"])
        for statement in statements
    ]

    bm25 = BM25Okapi(corpus)

    summaries: list[dict[str, Any]] = []

    for budget, result_path in sorted(
        RESULT_PATHS.items()
    ):
        summary = audit_budget(
            budget=budget,
            result_path=result_path,
            questions=questions,
            answers=answers,
            statements=statements,
            bm25=bm25,
        )

        summaries.append(summary)

    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "dataset": DATASET_NAME,
                "split": SPLIT_NAME,
                "source": SOURCE_NAME,
                "diagnostic_warning": (
                    "Gold answers were used to identify "
                    "oracle statements."
                ),
                "budgets": summaries,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    csv_fields = [
        "budget",
        "question_count",
        "correct_count",
        "accuracy_percent",
        "oracle_present_count",
        "oracle_recall_percent",
        "oracle_present_correct",
        "accuracy_when_oracle_present_percent",
        "oracle_missing_count",
        "oracle_missing_correct",
        "accuracy_when_oracle_missing_percent",
        "mean_statement_count",
        "mean_evidence_tokens",
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

        for summary in summaries:
            writer.writerow(
                {
                    field: summary.get(field)
                    for field in csv_fields
                }
            )

    print()
    print("=" * 94)
    print("BM25 BUDGET AUDIT")
    print("=" * 94)
    print(
        f"{'Budget':>8} "
        f"{'Acc.':>8} "
        f"{'Recall':>10} "
        f"{'Acc|present':>14} "
        f"{'Missing':>9} "
        f"{'Mean stmts':>12}"
    )

    for summary in summaries:
        print(
            f"{summary['budget']:>8} "
            f"{summary['accuracy_percent']:>7.1f}% "
            f"{format_optional(summary['oracle_recall_percent']):>9}% "
            f"{format_optional(summary['accuracy_when_oracle_present_percent']):>13}% "
            f"{summary['oracle_missing_count']:>9} "
            f"{summary['mean_statement_count']:>12.1f}"
        )

    print()
    print(
        "Important: oracle recall is diagnostic because "
        "gold answers were used to identify the supporting "
        "statement."
    )

    print(f"\nJSON summary: {OUTPUT_JSON}")
    print(f"CSV summary:  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()