from __future__ import annotations
import os

import json
import re
import statistics
import time
from getpass import getpass
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"

TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

QUESTION_LIMIT = 100

# This is the maximum number of tokens allocated only to
# retrieved memory statements.
EVIDENCE_TOKEN_BUDGET = int(
    os.environ.get("EVIDENCE_TOKEN_BUDGET", "128")
)

# The complete request is much smaller than 4K.
NUM_CTX = 4096

OUTPUT_PATH = Path(
    "results/"
    f"11_bm25_sh6k_budget"
    f"{EVIDENCE_TOKEN_BUDGET}.jsonl"
)

SUMMARY_PATH = Path(
    "results/"
    f"11_bm25_sh6k_budget"
    f"{EVIDENCE_TOKEN_BUDGET}_summary.json"
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
        number = int(match.group(1))

        text = (
            match.group(2)
            .strip()
            .replace("\n", " ")
        )

        statements.append(
            {
                "number": number,
                "text": text,
            }
        )

    return statements


def lexical_tokens(text: str) -> list[str]:
    return re.findall(
        r"\w+",
        text.casefold(),
        flags=re.UNICODE,
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


def count_tokens(
    tokenizer: Any,
    text: str,
) -> int:
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )


def retrieve_evidence(
    question: str,
    statements: list[dict[str, Any]],
    bm25: BM25Okapi,
    tokenizer: Any,
) -> tuple[str, list[dict[str, Any]], int]:
    query_tokens = lexical_tokens(question)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = sorted(
        range(len(statements)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    used_tokens = 0

    # Fifty candidates are more than enough for a 512-token budget.
    for index in ranked_indices[:50]:
        statement = statements[index]

        formatted = (
            f"{statement['number']}. "
            f"{statement['text']}"
        )

        statement_tokens = count_tokens(
            tokenizer,
            formatted,
        )

        if (
            used_tokens + statement_tokens
            > EVIDENCE_TOKEN_BUDGET
        ):
            continue

        selected.append(
            {
                "number": statement["number"],
                "text": statement["text"],
                "bm25_score": float(scores[index]),
                "token_count": statement_tokens,
            }
        )

        used_tokens += statement_tokens

    if not selected:
        raise RuntimeError(
            f"No evidence selected for question: {question}"
        )

    # The retriever chooses relevance. The prompt preserves the
    # original chronological ordering so that higher numbers appear
    # after older statements.
    selected.sort(
        key=lambda item: item["number"]
    )

    evidence = "\n".join(
        f"{item['number']}. {item['text']}"
        for item in selected
    )

    exact_evidence_tokens = count_tokens(
        tokenizer,
        evidence,
    )

    return evidence, selected, exact_evidence_tokens


def ask_model(
    session: requests.Session,
    evidence: str,
    question: str,
) -> dict[str, Any]:
    system_prompt = (
        "You are answering a synthetic memory benchmark. "
        "Use only the supplied memory evidence. "
        "The statements are chronological. Higher statement "
        "numbers are newer and replace conflicting facts with "
        "lower statement numbers. Ignore ordinary world knowledge "
        "when it conflicts with the supplied evidence. "
        "If the answer is not present in the evidence, answer "
        "UNKNOWN. Return only the shortest possible answer phrase."
    )

    user_prompt = (
        "RETRIEVED MEMORY EVIDENCE\n"
        "=========================\n"
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

    context = str(row["context"])
    questions = row["questions"]
    answers = row["answers"]

    statements = parse_numbered_statements(context)

    if not statements:
        raise RuntimeError(
            "No numbered statements were parsed."
        )

    print(
        f"Parsed {len(statements)} statements."
    )

    print(
        f"Loading tokenizer: {TOKENIZER_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        use_fast=True,
    )

    corpus = [
        lexical_tokens(statement["text"])
        for statement in statements
    ]

    bm25 = BM25Okapi(corpus)

    print(f"Model: {MODEL}")
    print(
        "Evidence-token budget:",
        EVIDENCE_TOKEN_BUDGET,
    )
    print(
        "Questions:",
        QUESTION_LIMIT,
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
        for question_index in range(QUESTION_LIMIT):
            question = str(
                questions[question_index]
            )

            gold_answers = as_gold_list(
                answers[question_index]
            )

            (
                evidence,
                retrieved_statements,
                evidence_tokens,
            ) = retrieve_evidence(
                question=question,
                statements=statements,
                bm25=bm25,
                tokenizer=tokenizer,
            )

            model_result = ask_model(
                session=session,
                evidence=evidence,
                question=question,
            )

            prediction = model_result["model_answer"]

            correct = substring_exact_match(
                prediction=prediction,
                gold_answers=gold_answers,
            )

            result = {
                "source": SOURCE_NAME,
                "condition": "bm25",
                "question_index": question_index,
                "question": question,
                "gold_answers": gold_answers,
                "model_answer": prediction,
                "substring_exact_match": correct,
                "model": MODEL,
                "requested_num_ctx": NUM_CTX,
                "evidence_token_budget":
                    EVIDENCE_TOKEN_BUDGET,
                "evidence_token_count":
                    evidence_tokens,
                "retrieved_statement_count":
                    len(retrieved_statements),
                "retrieved_statements":
                    retrieved_statements,
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
                f"{'CORRECT' if correct else 'WRONG'} "
                f"| evidence={evidence_tokens} tokens "
                f"| statements={len(retrieved_statements)}"
            )

            if question_index < 5:
                print("  Question:", question)
                print("  Gold:", gold_answers)
                print("  Answer:", prediction)
                print(
                    "  Retrieved numbers:",
                    [
                        item["number"]
                        for item in retrieved_statements
                    ],
                )

    correct_count = sum(
        int(result["substring_exact_match"])
        for result in results
    )

    evidence_counts = [
        result["evidence_token_count"]
        for result in results
    ]

    prompt_counts = [
        result["prompt_eval_count"]
        for result in results
        if result["prompt_eval_count"] is not None
    ]

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "condition": "bm25",
        "model": MODEL,
        "question_count": len(results),
        "correct_count": correct_count,
        "substring_exact_match_percent": (
            100 * correct_count / len(results)
        ),
        "evidence_token_budget":
            EVIDENCE_TOKEN_BUDGET,
        "minimum_evidence_tokens":
            min(evidence_counts),
        "maximum_evidence_tokens":
            max(evidence_counts),
        "mean_evidence_tokens":
            statistics.mean(evidence_counts),
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
    print("BM25 SUMMARY")
    print(
        f"Correct: {correct_count}/{len(results)}"
    )
    print(
        "Substring exact match:",
        f"{summary['substring_exact_match_percent']:.1f}%",
    )
    print(
        "Evidence-token range:",
        summary["minimum_evidence_tokens"],
        "to",
        summary["maximum_evidence_tokens"],
    )
    print(
        "Mean evidence tokens:",
        f"{summary['mean_evidence_tokens']:.1f}",
    )
    print(
        "Prompt-token range:",
        summary["minimum_prompt_eval_count"],
        "to",
        summary["maximum_prompt_eval_count"],
    )
    print(f"Detailed results: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()