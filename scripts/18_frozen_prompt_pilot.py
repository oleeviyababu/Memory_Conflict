from __future__ import annotations

import hashlib
import json
import re
import os
import time
from collections import defaultdict
from getpass import getpass
from math import comb
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

CHAIN_PATH = Path(
    "results/14_update_chains.jsonl"
)

OUTPUT_PATH = Path(
    "results/18_frozen_prompt_pilot.jsonl"
)

SUMMARY_PATH = Path(
    "results/18_frozen_prompt_pilot_summary.json"
)

BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"

NUM_CTX = 2048

STOPWORDS = {
    "a", "an", "and", "at", "by", "did", "do", "does",
    "for", "from", "in", "is", "of", "on", "the", "to",
    "was", "were", "what", "where", "which", "who",
    "whose", "with",
}


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
                "text": " ".join(
                    match.group(2)
                    .strip()
                    .replace("\n", " ")
                    .split()
                ),
            }
        )

    return statements


def lexical_tokens(text: str) -> set[str]:
    tokens = re.findall(
        r"\w+",
        text.casefold(),
        flags=re.UNICODE,
    )

    return {
        token
        for token in tokens
        if token not in STOPWORDS
        and len(token) > 1
    }


def normalize(text: str) -> str:
    text = text.casefold().strip()

    text = re.sub(
        r"[^\w\s-]",
        " ",
        text,
        flags=re.UNICODE,
    )

    return re.sub(r"\s+", " ", text).strip()


def substring_exact_match(
    prediction: str,
    gold_answers: list[str],
) -> bool:
    normalized_prediction = normalize(prediction)

    return any(
        normalize(gold) in normalized_prediction
        for gold in gold_answers
        if normalize(gold)
    )


def deterministic_key(
    question_index: int,
    statement_number: int,
) -> str:
    value = (
        f"{question_index}:"
        f"{statement_number}:"
        "controlled-distractor-v1"
    )

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def select_unrelated_distractors(
    question_index: int,
    question: str,
    gold_answers: list[str],
    forbidden_numbers: set[int],
    forbidden_values: list[str],
    statements: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    question_tokens = lexical_tokens(question)

    blocked_values = [
        normalize(value)
        for value in (
            list(gold_answers)
            + forbidden_values
        )
        if normalize(value)
    ]

    candidates: list[
        tuple[str, dict[str, Any]]
    ] = []

    for statement in statements:
        number = int(statement["number"])

        if number in forbidden_numbers:
            continue

        statement_text = str(statement["text"])
        normalized_statement = normalize(
            statement_text
        )

        if any(
            value in normalized_statement
            for value in blocked_values
        ):
            continue

        statement_tokens = lexical_tokens(
            statement_text
        )

        # Require zero content-word overlap with the question.
        if question_tokens & statement_tokens:
            continue

        candidates.append(
            (
                deterministic_key(
                    question_index,
                    number,
                ),
                statement,
            )
        )

    candidates.sort(
        key=lambda item: item[0]
    )

    selected = [
        dict(item[1])
        for item in candidates[:count]
    ]

    if len(selected) != count:
        raise RuntimeError(
            f"Could select only {len(selected)} "
            f"unrelated distractors for question "
            f"{question_index}; required {count}."
        )

    return selected


def format_evidence(
    statements: list[dict[str, Any]],
) -> str:
    chronological = sorted(
        statements,
        key=lambda item: int(item["number"]),
    )

    return "\n".join(
        f"{item['number']}. {item['text']}"
        for item in chronological
    )


def ask_model(
    session: requests.Session,
    question: str,
    evidence: str,
) -> dict[str, Any]:
    system_prompt = (
    "You are answering a synthetic memory benchmark. "
    "Use only the supplied numbered memory statements as "
    "the source of truth, even when they contradict ordinary "
    "world knowledge. The statements are chronological: "
    "higher statement numbers are newer and replace lower "
    "conflicting statements about the same fact. "
    "Return only the shortest possible answer phrase. "
    "Do not explain."
)

    user_prompt = (
        "MEMORY EVIDENCE\n"
        "===============\n"
        f"{evidence}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}\n\n"
        "ANSWER"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": 0,
            "seed": 42,
            "num_predict": 64,
        },
    }

    start = time.perf_counter()

    response = session.post(
        f"{BASE_URL}/api/chat",
        json=payload,
        timeout=600,
    )

    elapsed = time.perf_counter() - start

    response.raise_for_status()
    data = response.json()

    return {
        "model_answer": (
            data.get("message", {})
            .get("content", "")
            .strip()
        ),
        "prompt_eval_count":
            data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "client_elapsed_seconds": elapsed,
    }


def exact_mcnemar_p_value(
    first_only_correct: int,
    second_only_correct: int,
) -> float:
    discordant = (
        first_only_correct
        + second_only_correct
    )

    if discordant == 0:
        return 1.0

    smaller = min(
        first_only_correct,
        second_only_correct,
    )

    lower_tail = sum(
        comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2 * lower_tail)


def paired_comparison(
    first_name: str,
    second_name: str,
    results_by_condition: dict[
        str,
        dict[int, dict[str, Any]],
    ],
    indices: list[int],
) -> dict[str, Any]:
    first_only_correct = 0
    second_only_correct = 0
    both_correct = 0
    both_wrong = 0

    for index in indices:
        first = results_by_condition[
            first_name
        ].get(index)

        second = results_by_condition[
            second_name
        ].get(index)

        if first is None or second is None:
            continue

        first_correct = bool(
            first["substring_exact_match"]
        )

        second_correct = bool(
            second["substring_exact_match"]
        )

        if first_correct and second_correct:
            both_correct += 1
        elif first_correct:
            first_only_correct += 1
        elif second_correct:
            second_only_correct += 1
        else:
            both_wrong += 1

    total = (
        both_correct
        + first_only_correct
        + second_only_correct
        + both_wrong
    )

    return {
        "first_condition": first_name,
        "second_condition": second_name,
        "question_count": total,
        "both_correct": both_correct,
        "first_only_correct":
            first_only_correct,
        "second_only_correct":
            second_only_correct,
        "both_wrong": both_wrong,
        "exact_mcnemar_p_value":
            exact_mcnemar_p_value(
                first_only_correct,
                second_only_correct,
            ),
    }


def main() -> None:
    if not CHAIN_PATH.exists():
        raise FileNotFoundError(
            f"Missing chain file: {CHAIN_PATH}"
        )

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    statements = parse_numbered_statements(
        str(row["context"])
    )

    chain_rows = sorted(
        load_jsonl(CHAIN_PATH),
        key=lambda item: int(
            item["question_index"]
        ),
    )

    chain_rows = [
        row
        for row in chain_rows
        if int(row["previous_conflict_count"]) > 0
    ][:10]

    expected_calls = len(chain_rows) * 4

    print(
        f"Parsed {len(statements)} statements."
    )
    print(
        f"Loaded {len(chain_rows)} update chains."
    )
    print(
        f"Planned calls: {len(chain_rows)} questions "
        f"x 4 conditions = {expected_calls} calls"
    )

    session = requests.Session()
    
    if USERNAME:
        password = getpass("\nOllama password: ")
        session.auth = (USERNAME, password)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results: list[dict[str, Any]] = []

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        call_number = 0

        for chain_row in chain_rows:
            question_index = int(
                chain_row["question_index"]
            )

            question = str(
                chain_row["question"]
            )

            gold_answers = [
                str(answer)
                for answer in chain_row[
                    "gold_answers"
                ]
            ]

            latest = {
                "number": int(
                    chain_row["oracle_number"]
                ),
                "text": str(
                    chain_row["oracle_text"]
                ),
            }

            previous_conflicts = chain_row.get(
                "previous_conflicting_statements",
                [],
            )

            conflict = (
                dict(previous_conflicts[-1])
                if previous_conflicts
                else None
            )

            chain_numbers = {
                int(item["number"])
                for item in chain_row["chain"]
            }

            forbidden_values = [
                str(item.get("value", ""))
                for item in chain_row["chain"]
            ]

            distractors = (
                select_unrelated_distractors(
                    question_index=question_index,
                    question=question,
                    gold_answers=gold_answers,
                    forbidden_numbers=chain_numbers,
                    forbidden_values=forbidden_values,
                    statements=statements,
                    count=3,
                )
            )

            conditions: list[
                tuple[str, list[dict[str, Any]]]
            ] = [
                (
                    "latest_only",
                    [latest],
                ),
                (
                    "latest_plus_3_unrelated",
                    [latest] + distractors,
                ),
            ]

            group = (
                "conflict"
                if conflict is not None
                else "no_conflict"
            )

            if conflict is not None:
                conditions.extend(
                    [
                        (
                            "old_plus_latest",
                            [conflict, latest],
                        ),
                        (
                            "old_plus_latest_plus_2_unrelated",
                            [
                                conflict,
                                latest,
                                distractors[0],
                                distractors[1],
                            ],
                        ),
                    ]
                )

            for condition, evidence_items in conditions:
                call_number += 1

                evidence = format_evidence(
                    evidence_items
                )

                model_result = ask_model(
                    session=session,
                    question=question,
                    evidence=evidence,
                )

                correct = substring_exact_match(
                    prediction=model_result[
                        "model_answer"
                    ],
                    gold_answers=gold_answers,
                )

                result = {
                    "question_index":
                        question_index,
                    "question": question,
                    "gold_answers":
                        gold_answers,
                    "group": group,
                    "condition": condition,
                    "statement_count":
                        len(evidence_items),
                    "evidence_statement_numbers":
                        sorted(
                            int(item["number"])
                            for item
                            in evidence_items
                        ),
                    "evidence_statements":
                        sorted(
                            evidence_items,
                            key=lambda item:
                                int(item["number"]),
                        ),
                    "model": MODEL,
                    "requested_num_ctx":
                        NUM_CTX,
                    "substring_exact_match":
                        correct,
                    **model_result,
                }

                all_results.append(result)

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()

                print(
                    f"{call_number:3}/{expected_calls} "
                    f"| q={question_index + 1:3} "
                    f"| {condition:<36} "
                    f"| "
                    f"{'CORRECT' if correct else 'WRONG'}"
                )

    results_by_condition: dict[
        str,
        dict[int, dict[str, Any]],
    ] = defaultdict(dict)

    for result in all_results:
        results_by_condition[
            result["condition"]
        ][int(result["question_index"])] = (
            result
        )

    conflict_indices = sorted(
        int(row["question_index"])
        for row in chain_rows
        if int(
            row["previous_conflict_count"]
        ) > 0
    )

    no_conflict_indices = sorted(
        int(row["question_index"])
        for row in chain_rows
        if int(
            row["previous_conflict_count"]
        ) == 0
    )

    condition_summaries = []

    for condition, indexed in (
        results_by_condition.items()
    ):
        for group_name, indices in [
            ("all", sorted(indexed)),
            ("conflict", conflict_indices),
            ("no_conflict", no_conflict_indices),
        ]:
            available = [
                index
                for index in indices
                if index in indexed
            ]

            if not available:
                continue

            correct_count = sum(
                int(
                    indexed[index][
                        "substring_exact_match"
                    ]
                )
                for index in available
            )

            condition_summaries.append(
                {
                    "condition": condition,
                    "group": group_name,
                    "question_count":
                        len(available),
                    "correct_count":
                        correct_count,
                    "accuracy_percent": (
                        100
                        * correct_count
                        / len(available)
                    ),
                }
            )

    paired_comparisons = [
        paired_comparison(
            "latest_only",
            "latest_plus_3_unrelated",
            results_by_condition,
            conflict_indices,
        ),
        paired_comparison(
            "latest_only",
            "old_plus_latest",
            results_by_condition,
            conflict_indices,
        ),
        paired_comparison(
            "latest_plus_3_unrelated",
            "old_plus_latest_plus_2_unrelated",
            results_by_condition,
            conflict_indices,
        ),
        paired_comparison(
            "old_plus_latest",
            "old_plus_latest_plus_2_unrelated",
            results_by_condition,
            conflict_indices,
        ),
    ]

    summary = {
        "model": MODEL,
        "source": SOURCE_NAME,
        "call_count": len(all_results),
        "condition_summaries":
            condition_summaries,
        "paired_comparisons":
            paired_comparisons,
        "distractor_selection": (
            "Deterministic statements with zero "
            "content-word overlap with the question, "
            "excluding update-chain values."
        ),
        "result_file": str(OUTPUT_PATH),
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
    print("=" * 96)
    print("CONTROLLED CONDITION SUMMARY")
    print("=" * 96)

    for item in condition_summaries:
        print(
            f"{item['condition']:<38} "
            f"{item['group']:<12} "
            f"{item['correct_count']:>3}/"
            f"{item['question_count']:<3} "
            f"({item['accuracy_percent']:.1f}%)"
        )

    print()
    print("=" * 96)
    print("PAIRED COMPARISONS")
    print("=" * 96)

    for comparison in paired_comparisons:
        print(
            f"{comparison['first_condition']} "
            f"vs "
            f"{comparison['second_condition']} "
            f"| first-only="
            f"{comparison['first_only_correct']} "
            f"| second-only="
            f"{comparison['second_only_correct']} "
            f"| p="
            f"{comparison['exact_mcnemar_p_value']:.4f}"
        )

    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
