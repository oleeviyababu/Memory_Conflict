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

NUM_CTX = 4096
QUESTION_LIMIT = 100

OUTPUT_PATH = Path(
    "results/06_question_only_sh6k_all.jsonl"
)
SUMMARY_PATH = Path(
    "results/06_question_only_sh6k_all_summary.json"
)


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}

        if metadata.get("source") == SOURCE_NAME:
            return row

    raise RuntimeError(
        f"Could not find source: {SOURCE_NAME}"
    )


def as_gold_list(answer: Any) -> list[str]:
    if isinstance(answer, list):
        return [str(item) for item in answer]

    return [str(answer)]


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
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Answer the question using your existing "
                    "general knowledge. Return only the shortest "
                    "possible answer phrase. Do not explain."
                ),
            },
            {
                "role": "user",
                "content": question,
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
        "client_elapsed_seconds": elapsed,
        "server_total_seconds": (
            data.get("total_duration", 0)
            / 1_000_000_000
        ),
    }


def main() -> None:
    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )

    row = find_dataset_row(dataset)

    questions = row["questions"]
    answers = row["answers"]

    print(f"Selected source: {SOURCE_NAME}")
    print(f"Running {QUESTION_LIMIT} questions")
    print(f"Condition: question only")
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

            model_result = ask_model(
                session=session,
                question=question,
            )

            prediction = model_result["model_answer"]

            correct = substring_exact_match(
                prediction=prediction,
                gold_answers=gold_answers,
            )

            result = {
                "source": SOURCE_NAME,
                "condition": "question_only",
                "question_index": question_index,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": prediction,
                "substring_exact_match": correct,
                "model": MODEL,
                "requested_num_ctx": NUM_CTX,
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

            print(
                f"{question_index + 1:3}/"
                f"{QUESTION_LIMIT}: "
                f"{'CORRECT' if correct else 'WRONG'}"
            )
            print("  Question:", question)
            print("  Gold:", gold_answers)
            print("  Answer:", prediction)

    correct_count = sum(
        int(result["substring_exact_match"])
        for result in results
    )

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "condition": "question_only",
        "model": MODEL,
        "question_count": len(results),
        "correct_count": correct_count,
        "substring_exact_match_percent": (
            100 * correct_count / len(results)
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
    print("QUESTION-ONLY SUMMARY")
    print(f"Correct: {correct_count}/{len(results)}")
    print(
        "Substring exact match:",
        f"{summary['substring_exact_match_percent']:.1f}%",
    )
    print(f"Detailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()