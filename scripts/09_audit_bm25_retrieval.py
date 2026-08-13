from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset
from rank_bm25 import BM25Okapi


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

BM25_RESULTS_PATH = Path(
    "results/08_bm25_sh6k_budget512.jsonl"
)

OUTPUT_PATH = Path(
    "results/09_bm25_retrieval_audit.jsonl"
)

SUMMARY_PATH = Path(
    "results/09_bm25_retrieval_audit_summary.json"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

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

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def as_gold_list(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(item) for item in answer]

    return [str(answer)]


def contains_any_gold(
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
    Diagnostic oracle selection.

    First, retain only statements containing a gold answer.
    Then choose the candidate with the highest BM25 similarity
    to the question. A higher statement number breaks score ties.

    Because this uses the gold answer, it is an analysis oracle,
    not a deployable retrieval method.
    """
    all_scores = bm25.get_scores(
        lexical_tokens(question)
    )

    candidates: list[dict[str, Any]] = []

    for index, statement in enumerate(statements):
        if not contains_any_gold(
            statement["text"],
            gold_answers,
        ):
            continue

        candidates.append(
            {
                "number": statement["number"],
                "text": statement["text"],
                "question_bm25_score": float(
                    all_scores[index]
                ),
            }
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["question_bm25_score"],
            item["number"],
        ),
        reverse=True,
    )

    selected = dict(candidates[0])
    selected["candidate_count"] = len(candidates)
    selected["top_candidates"] = candidates[:5]

    return selected


def main() -> None:
    if not BM25_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing result file: {BM25_RESULTS_PATH}"
        )

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    context = str(row["context"])
    questions = row["questions"]
    answers = row["answers"]

    statements = parse_numbered_statements(context)

    print(f"Parsed {len(statements)} statements.")

    corpus = [
        lexical_tokens(statement["text"])
        for statement in statements
    ]

    question_ranker = BM25Okapi(corpus)

    bm25_results = {
        int(result["question_index"]): result
        for result in load_jsonl(
            BM25_RESULTS_PATH
        )
    }

    categories: Counter[str] = Counter()
    audit_rows: list[dict[str, Any]] = []

    resolved_count = 0
    oracle_retrieved_count = 0
    gold_anywhere_count = 0

    oracle_present_total = 0
    oracle_present_correct = 0

    oracle_missing_total = 0
    oracle_missing_correct = 0

    for question_index in range(len(questions)):
        question = str(questions[question_index])
        gold_answers = as_gold_list(
            answers[question_index]
        )

        bm25_result = bm25_results.get(
            question_index
        )

        if bm25_result is None:
            raise RuntimeError(
                "Missing BM25 result for question "
                f"{question_index}"
            )

        selected_oracle = (
            select_gold_assisted_statement(
                question=question,
                gold_answers=gold_answers,
                statements=statements,
                bm25=question_ranker,
            )
        )

        retrieved_statements = bm25_result[
            "retrieved_statements"
        ]

        retrieved_numbers = {
            int(statement["number"])
            for statement in retrieved_statements
        }

        retrieved_text = "\n".join(
            str(statement["text"])
            for statement in retrieved_statements
        )

        gold_anywhere_in_retrieval = (
            contains_any_gold(
                retrieved_text,
                gold_answers,
            )
        )

        if gold_anywhere_in_retrieval:
            gold_anywhere_count += 1

        bm25_correct = bool(
            bm25_result["substring_exact_match"]
        )

        if selected_oracle is None:
            category = "oracle_unresolved"
            oracle_statement_retrieved = None

        else:
            resolved_count += 1

            oracle_number = int(
                selected_oracle["number"]
            )

            oracle_statement_retrieved = (
                oracle_number in retrieved_numbers
            )

            if oracle_statement_retrieved:
                oracle_retrieved_count += 1
                oracle_present_total += 1

                if bm25_correct:
                    oracle_present_correct += 1
                    category = (
                        "oracle_present_answer_correct"
                    )
                else:
                    category = (
                        "oracle_present_answer_wrong"
                    )

            else:
                oracle_missing_total += 1

                if bm25_correct:
                    oracle_missing_correct += 1
                    category = (
                        "oracle_missing_answer_correct"
                    )
                else:
                    category = (
                        "oracle_missing_answer_wrong"
                    )

        categories[category] += 1

        audit = {
            "question_index": question_index,
            "question": question,
            "gold_answers": gold_answers,
            "bm25_answer":
                bm25_result["model_answer"],
            "bm25_correct": bm25_correct,
            "retrieved_statement_count": len(
                retrieved_statements
            ),
            "retrieved_numbers": sorted(
                retrieved_numbers
            ),
            "gold_answer_string_anywhere_in_retrieval":
                gold_anywhere_in_retrieval,
            "gold_assisted_oracle_statement":
                selected_oracle,
            "oracle_statement_retrieved":
                oracle_statement_retrieved,
            "category": category,
        }

        audit_rows.append(audit)

        if question_index < 10:
            print()
            print("=" * 72)
            print(
                f"QUESTION {question_index + 1}"
            )
            print("Question:", question)
            print("Gold:", gold_answers)
            print(
                "BM25 answer:",
                bm25_result["model_answer"],
            )
            print("BM25 correct:", bm25_correct)

            if selected_oracle is None:
                print(
                    "Oracle statement: NOT RESOLVED"
                )
            else:
                print(
                    "Oracle statement:",
                    f"{selected_oracle['number']}. "
                    f"{selected_oracle['text']}",
                )
                print(
                    "Oracle statement retrieved:",
                    oracle_statement_retrieved,
                )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for audit in audit_rows:
            output_file.write(
                json.dumps(
                    audit,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(audit_rows)

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "question_count": total,
        "oracle_resolved_count": resolved_count,
        "oracle_unresolved_count": (
            total - resolved_count
        ),
        "oracle_statement_retrieved_count":
            oracle_retrieved_count,
        "oracle_statement_recall_percent": (
            100
            * oracle_retrieved_count
            / resolved_count
            if resolved_count
            else None
        ),
        "gold_answer_string_anywhere_count":
            gold_anywhere_count,
        "gold_answer_string_anywhere_percent": (
            100 * gold_anywhere_count / total
        ),
        "oracle_present_total":
            oracle_present_total,
        "oracle_present_correct":
            oracle_present_correct,
        "accuracy_when_oracle_present_percent": (
            100
            * oracle_present_correct
            / oracle_present_total
            if oracle_present_total
            else None
        ),
        "oracle_missing_total":
            oracle_missing_total,
        "oracle_missing_correct":
            oracle_missing_correct,
        "accuracy_when_oracle_missing_percent": (
            100
            * oracle_missing_correct
            / oracle_missing_total
            if oracle_missing_total
            else None
        ),
        "categories": dict(categories),
        "audit_file": str(OUTPUT_PATH),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BM25 RETRIEVAL AUDIT")
    print(
        "Oracle statements resolved:",
        f"{resolved_count}/{total}",
    )
    print(
        "Oracle statement retrieved:",
        f"{oracle_retrieved_count}/"
        f"{resolved_count}",
    )
    print(
        "Oracle-statement recall:",
        (
            f"{summary['oracle_statement_recall_percent']:.1f}%"
            if summary[
                "oracle_statement_recall_percent"
            ] is not None
            else "N/A"
        ),
    )
    print(
        "Gold answer string somewhere in evidence:",
        f"{gold_anywhere_count}/{total}",
    )
    print()
    print(
        "Accuracy when oracle statement was present:",
        f"{oracle_present_correct}/"
        f"{oracle_present_total}",
    )
    print(
        "Accuracy when oracle statement was missing:",
        f"{oracle_missing_correct}/"
        f"{oracle_missing_total}",
    )
    print()
    print("Categories:")

    for category, count in sorted(
        categories.items()
    ):
        print(f"  {category}: {count}")

    print()
    print(f"Detailed audit: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()