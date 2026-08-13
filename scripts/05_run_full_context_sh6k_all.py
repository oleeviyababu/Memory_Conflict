from __future__ import annotations

import json
import re
import time
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

# Use a comfortably large window for this first full-context test.
NUM_CTX = 16384
QUESTION_LIMIT = 100


OUTPUT_PATH = Path(
    "results/05_full_context_sh6k_all.jsonl"
)
SUMMARY_PATH = Path(
    "results/05_full_context_sh6k_summary.json"
)


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    """Select the requested FactConsolidation history."""
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}

        if metadata.get("source") == SOURCE_NAME:
            return row

    raise RuntimeError(
        f"Could not find dataset source: {SOURCE_NAME}"
    )


def as_gold_list(answer: Any) -> list[str]:
    """Convert the benchmark answer into a list of strings."""
    if isinstance(answer, list):
        return [str(item) for item in answer]

    return [str(answer)]


def normalize(text: str) -> str:
    """
    Normalize text for case-insensitive substring scoring.
    Unicode letters such as ä are retained.
    """
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def substring_exact_match(
    prediction: str,
    gold_answers: list[str],
) -> bool:
    """
    Mark the response correct when a normalized gold answer
    occurs in the normalized model response.
    """
    normalized_prediction = normalize(prediction)

    for gold in gold_answers:
        normalized_gold = normalize(gold)

        if normalized_gold and normalized_gold in normalized_prediction:
            return True

    return False


def ask_model(
    session: requests.Session,
    context: str,
    question: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are answering a synthetic memory benchmark. "
        "Treat the supplied memory log as the only source of truth, "
        "even when it contradicts ordinary world knowledge. "
        "The memory log is chronological, so later statements may "
        "update or replace earlier statements. "
        "Answer with only the shortest possible answer phrase. "
        "Do not provide an explanation."
    )

    user_prompt = (
        "MEMORY LOG\n"
        "==========\n"
        f"{context}\n\n"
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

    start_time = time.perf_counter()

    response = session.post(
        f"{BASE_URL}/api/chat",
        json=payload,
        timeout=600,
    )

    elapsed_seconds = time.perf_counter() - start_time
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
        "server_total_seconds": (
            data.get("total_duration", 0)
            / 1_000_000_000
        ),
        "client_elapsed_seconds": elapsed_seconds,
    }


def main() -> None:
    print(
        f"Loading {DATASET_NAME}, "
        f"split={SPLIT_NAME}..."
    )

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    context = row["context"]
    questions = row["questions"]
    answers = row["answers"]

    print(f"Selected source: {SOURCE_NAME}")
    print(f"Context characters: {len(context):,}")
    print(f"Context words: {len(context.split()):,}")
    print(f"Available questions: {len(questions)}")
    print(f"Running first {QUESTION_LIMIT} questions")
    print(f"Requested num_ctx: {NUM_CTX}")
    print(f"Model: {MODEL}")

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
        for question_index in range(QUESTION_LIMIT):
            question = str(questions[question_index])
            gold_answers = as_gold_list(
                answers[question_index]
            )

            print()
            print("-" * 72)
            print(
                f"Question "
                f"{question_index + 1}/{QUESTION_LIMIT}"
            )
            print("Question:", question)
            print("Gold:", gold_answers)

            model_result = ask_model(
                session=session,
                context=context,
                question=question,
            )

            prediction = model_result["model_answer"]

            is_correct = substring_exact_match(
                prediction=prediction,
                gold_answers=gold_answers,
            )

            result = {
                "source": SOURCE_NAME,
                "question_index": question_index,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": prediction,
                "substring_exact_match": is_correct,
                "requested_num_ctx": NUM_CTX,
                "model": MODEL,
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

            print("Model answer:", prediction)
            print("Correct:", is_correct)
            print(
                "Prompt tokens:",
                model_result["prompt_eval_count"],
            )
            print(
                "Generated tokens:",
                model_result["eval_count"],
            )
            print(
                "Time:",
                round(
                    model_result[
                        "client_elapsed_seconds"
                    ],
                    2,
                ),
                "seconds",
            )

    correct_count = sum(
        int(result["substring_exact_match"])
        for result in results
    )

    prompt_counts = [
        result["prompt_eval_count"]
        for result in results
        if result["prompt_eval_count"] is not None
    ]

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "model": MODEL,
        "requested_num_ctx": NUM_CTX,
        "question_count": len(results),
        "correct_count": correct_count,
        "substring_exact_match_percent": (
            100 * correct_count / len(results)
            if results
            else 0
        ),
        "minimum_prompt_eval_count": (
            min(prompt_counts)
            if prompt_counts
            else None
        ),
        "maximum_prompt_eval_count": (
            max(prompt_counts)
            if prompt_counts
            else None
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
    print("=" * 72)
    print("SMOKE-TEST SUMMARY")
    print(f"Correct: {correct_count}/{len(results)}")
    print(
        "Substring exact match:",
        f"{summary['substring_exact_match_percent']:.1f}%",
    )
    print(
        "Prompt-token range:",
        summary["minimum_prompt_eval_count"],
        "to",
        summary["maximum_prompt_eval_count"],
    )
    print(f"Detailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")

    if prompt_counts and min(prompt_counts) < 4000:
        print()
        print(
            "WARNING: The processed prompt appears unexpectedly "
            "short. Inspect for truncation before proceeding."
        )
    else:
        print()
        print(
            "Context check looks reasonable: every request "
            "processed more than 4,000 prompt tokens."
        )


if __name__ == "__main__":
    main()