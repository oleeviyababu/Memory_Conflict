from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from getpass import getpass
from pathlib import Path
from typing import Any
import os

import requests
from datasets import load_dataset


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"


BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"
NUM_CTX = 4096

SMOKE_RESULTS_PATH = Path(
    "results/02_full_context_sh6k_smoke.jsonl"
)

OUTPUT_PATH = Path(
    "results/04_oracle_diagnostic.jsonl"
)

SUMMARY_PATH = Path(
    "results/04_oracle_diagnostic_summary.json"
)


# We currently diagnose only the three failed smoke-test questions.
CASES = [
    {
        "question_index": 0,
        "relation_pattern": (
            "goaltender is associated with the sport of"
        ),
    },
    {
        "question_index": 1,
        "relation_pattern": (
            "Nobuhiro Watsuki is famous for"
        ),
    },
    {
        "question_index": 2,
        "relation_pattern": (
            "rugby union was created in the country of"
        ),
    },
]


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}

        if metadata.get("source") == SOURCE_NAME:
            return row

    raise RuntimeError(
        f"Could not find source: {SOURCE_NAME}"
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def parse_numbered_statements(
    context: str,
) -> list[dict[str, Any]]:
    """
    Parse text formatted approximately as:

    1. First statement. 2. Second statement. ...
    """
    pattern = re.compile(
        r"(?<!\d)(\d+)\.\s+"
        r"(.*?)"
        r"(?=\s+\d+\.\s+|\Z)",
        flags=re.DOTALL,
    )

    statements: list[dict[str, Any]] = []

    for match in pattern.finditer(context):
        statement_number = int(match.group(1))

        statement_text = (
            match.group(2)
            .strip()
            .replace("\n", " ")
        )

        statements.append(
            {
                "number": statement_number,
                "text": statement_text,
            }
        )

    return statements


def find_relevant_statements(
    statements: list[dict[str, Any]],
    relation_pattern: str,
) -> list[dict[str, Any]]:
    pattern = relation_pattern.casefold()

    matches = [
        statement
        for statement in statements
        if pattern in statement["text"].casefold()
    ]

    return sorted(
        matches,
        key=lambda item: item["number"],
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


def ask_model(
    session: requests.Session,
    question: str,
    condition: str,
    evidence: str | None,
) -> dict[str, Any]:
    if condition == "question_only":
        system_prompt = (
            "Answer the question using your general knowledge. "
            "Return only the shortest possible answer phrase. "
            "Do not explain."
        )

        user_prompt = question

    else:
        system_prompt = (
            "You are answering a synthetic memory benchmark. "
            "Treat the supplied numbered statements as the only "
            "source of truth, even when they contradict ordinary "
            "world knowledge. Higher statement numbers represent "
            "newer information and replace conflicting statements "
            "with lower numbers. Return only the shortest possible "
            "answer phrase. Do not explain."
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
        "prompt_eval_count": data.get(
            "prompt_eval_count"
        ),
        "eval_count": data.get("eval_count"),
        "elapsed_seconds": elapsed,
    }


def format_statements(
    statements: list[dict[str, Any]],
) -> str:
    return "\n".join(
        f"{statement['number']}. {statement['text']}"
        for statement in statements
    )


def main() -> None:
    if not SMOKE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {SMOKE_RESULTS_PATH}"
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

    print(
        f"Parsed {len(statements)} numbered statements."
    )

    smoke_results = {
        int(result["question_index"]): result
        for result in load_jsonl(
            SMOKE_RESULTS_PATH
        )
    }

    session = requests.Session()

    if USERNAME:
        password = getpass("\nOllama password: ")
        session.auth = (USERNAME, password)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results: list[dict[str, Any]] = []

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for case in CASES:
            question_index = case["question_index"]
            relation_pattern = case[
                "relation_pattern"
            ]

            question = str(
                questions[question_index]
            )

            raw_gold = answers[question_index]

            gold_answers = (
                [str(item) for item in raw_gold]
                if isinstance(raw_gold, list)
                else [str(raw_gold)]
            )

            relevant = find_relevant_statements(
                statements=statements,
                relation_pattern=relation_pattern,
            )

            if not relevant:
                raise RuntimeError(
                    "No relevant statements found for "
                    f"pattern: {relation_pattern}"
                )

            latest_statement = relevant[-1]

            conditions = {
                "question_only": None,
                "latest_only": format_statements(
                    [latest_statement]
                ),
                "all_relevant_chronological":
                    format_statements(relevant),
            }

            print()
            print("=" * 80)
            print(
                f"QUESTION {question_index + 1}"
            )
            print("Question:", question)
            print("Gold:", gold_answers)

            print("\nRelevant statements:")

            for statement in relevant:
                print(
                    f"  {statement['number']}. "
                    f"{statement['text']}"
                )

            full_context_answer = (
                smoke_results
                .get(question_index, {})
                .get("model_answer")
            )

            print(
                "\nPrevious full-context answer:",
                full_context_answer,
            )

            for condition, evidence in conditions.items():
                model_result = ask_model(
                    session=session,
                    question=question,
                    condition=condition,
                    evidence=evidence,
                )

                is_correct = substring_exact_match(
                    prediction=model_result[
                        "model_answer"
                    ],
                    gold_answers=gold_answers,
                )

                result = {
                    "question_index": question_index,
                    "question": question,
                    "gold_answers": gold_answers,
                    "relation_pattern":
                        relation_pattern,
                    "relevant_statements":
                        relevant,
                    "latest_statement":
                        latest_statement,
                    "condition": condition,
                    "evidence": evidence,
                    "full_context_answer":
                        full_context_answer,
                    "model": MODEL,
                    "requested_num_ctx": NUM_CTX,
                    "correct": is_correct,
                    **model_result,
                }

                results.append(result)

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                print()
                print("Condition:", condition)
                print(
                    "Answer:",
                    model_result["model_answer"],
                )
                print("Correct:", is_correct)
                print(
                    "Prompt tokens:",
                    model_result[
                        "prompt_eval_count"
                    ],
                )

    condition_totals: dict[str, int] = (
        defaultdict(int)
    )

    condition_correct: dict[str, int] = (
        defaultdict(int)
    )

    for result in results:
        condition = result["condition"]

        condition_totals[condition] += 1

        if result["correct"]:
            condition_correct[condition] += 1

    summary = {
        "model": MODEL,
        "source": SOURCE_NAME,
        "question_count": len(CASES),
        "conditions": {},
        "result_file": str(OUTPUT_PATH),
    }

    for condition, total in condition_totals.items():
        correct = condition_correct[condition]

        summary["conditions"][condition] = {
            "correct": correct,
            "total": total,
            "accuracy_percent": (
                100 * correct / total
                if total
                else 0
            ),
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
    print("=" * 80)
    print("DIAGNOSTIC SUMMARY")

    for condition, values in summary[
        "conditions"
    ].items():
        print(
            f"{condition}: "
            f"{values['correct']}/"
            f"{values['total']} "
            f"({values['accuracy_percent']:.1f}%)"
        )

    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()