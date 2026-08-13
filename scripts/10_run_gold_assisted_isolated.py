from __future__ import annotations

import json
import re
import statistics
import time
from getpass import getpass
from pathlib import Path
from typing import Any
import os

import requests


AUDIT_PATH = Path(
    "results/09_bm25_retrieval_audit.jsonl"
)

OUTPUT_PATH = Path(
    "results/10_gold_assisted_isolated.jsonl"
)

SUMMARY_PATH = Path(
    "results/10_gold_assisted_isolated_summary.json"
)


BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"

NUM_CTX = 2048


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
                    f"Invalid JSON on line {line_number} "
                    f"of {path}"
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
    evidence: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are answering a synthetic memory benchmark. "
        "Use only the supplied memory statement as the source "
        "of truth, even when it contradicts ordinary world "
        "knowledge. Return only the shortest possible answer "
        "phrase. Do not explain."
    )

    user_prompt = (
        "MEMORY STATEMENT\n"
        "================\n"
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
        "client_elapsed_seconds": elapsed_seconds,
        "server_total_seconds": (
            data.get("total_duration", 0)
            / 1_000_000_000
        ),
    }


def main() -> None:
    if not AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing audit file: {AUDIT_PATH}"
        )

    audit_rows = load_jsonl(AUDIT_PATH)

    print(
        f"Loaded {len(audit_rows)} audited questions."
    )
    print(f"Model: {MODEL}")
    print("Condition: gold-assisted isolated statement")
    print()
    print(
        "WARNING: This condition uses gold answers to select "
        "evidence. It is a diagnostic upper bound, not a fair "
        "retrieval method."
    )

    

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
        for position, audit in enumerate(
            audit_rows,
            start=1,
        ):
            question_index = int(
                audit["question_index"]
            )

            question = str(audit["question"])

            gold_answers = [
                str(answer)
                for answer in audit["gold_answers"]
            ]

            oracle_statement = audit.get(
                "gold_assisted_oracle_statement"
            )

            if not oracle_statement:
                result = {
                    "question_index": question_index,
                    "question": question,
                    "gold_answers": gold_answers,
                    "condition":
                        "gold_assisted_isolated",
                    "skipped": True,
                    "skip_reason":
                        "No oracle statement available",
                }

                results.append(result)

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                print(
                    f"{position:3}/{len(audit_rows)}: "
                    "SKIPPED"
                )
                continue

            evidence = (
                f"{oracle_statement['number']}. "
                f"{oracle_statement['text']}"
            )

            model_result = ask_model(
                session=session,
                question=question,
                evidence=evidence,
            )

            prediction = model_result["model_answer"]

            correct = substring_exact_match(
                prediction=prediction,
                gold_answers=gold_answers,
            )

            result = {
                "question_index": question_index,
                "question": question,
                "gold_answers": gold_answers,
                "condition": "gold_assisted_isolated",
                "evidence_statement_number":
                    oracle_statement["number"],
                "evidence_statement_text":
                    oracle_statement["text"],
                "evidence": evidence,
                "model": MODEL,
                "requested_num_ctx": NUM_CTX,
                "model_answer": prediction,
                "substring_exact_match": correct,
                "skipped": False,
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
                f"{position:3}/{len(audit_rows)}: "
                f"{'CORRECT' if correct else 'WRONG'}"
            )

            if position <= 10 or not correct:
                print("  Question:", question)
                print("  Gold:", gold_answers)
                print("  Evidence:", evidence)
                print("  Answer:", prediction)

    evaluated_results = [
        result
        for result in results
        if not result.get("skipped", False)
    ]

    correct_count = sum(
        int(result["substring_exact_match"])
        for result in evaluated_results
    )

    prompt_counts = [
        result["prompt_eval_count"]
        for result in evaluated_results
        if result.get("prompt_eval_count") is not None
    ]

    elapsed_times = [
        result["client_elapsed_seconds"]
        for result in evaluated_results
    ]

    evaluated_count = len(evaluated_results)

    summary = {
        "condition": "gold_assisted_isolated",
        "diagnostic_only": True,
        "gold_answer_used_for_evidence_selection": True,
        "model": MODEL,
        "question_count": len(results),
        "evaluated_count": evaluated_count,
        "skipped_count": len(results) - evaluated_count,
        "correct_count": correct_count,
        "substring_exact_match_percent": (
            100 * correct_count / evaluated_count
            if evaluated_count
            else None
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
        "mean_client_elapsed_seconds": (
            statistics.mean(elapsed_times)
            if elapsed_times
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
    print("GOLD-ASSISTED ISOLATED-EVIDENCE SUMMARY")
    print(
        f"Correct: {correct_count}/{evaluated_count}"
    )

    if evaluated_count:
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

    print()
    print(
        "This result is a diagnostic upper bound because "
        "gold answers were used to select the statement."
    )

    print(f"\nDetailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()