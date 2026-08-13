from __future__ import annotations

import argparse
import json
import re
import os
import time
from collections import Counter
from getpass import getpass
from math import comb
from pathlib import Path
from statistics import mean
from typing import Any

import requests
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer



# Configuration

DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_32k"

BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"
TOKENIZER_NAME = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_PATH = Path("results/21b_sh32k_bm25_replication.jsonl")
SUMMARY_PATH = Path("results/21b_sh32k_bm25_replication_summary.json")
CHAIN_PATH = Path("results/21b_sh32k_update_chains.jsonl")
CHAIN_SUMMARY_PATH = Path("results/21b_sh32k_update_chains_summary.json")

PROMPT_VERSION = "frozen_non_abstaining_v1"
RETRIEVAL_VERSION = "bm25_okapi_regex_v1"
CHAIN_VERSION = "template_from_gold_statement_v1"

TEMPERATURE = 0
SEED = 42
NUM_PREDICT = 64
NUM_CTX = 2048

CONDITIONS = ["bm25_64", "bm25_512"]

STOPWORDS = {
    "a", "an", "and", "at", "by", "did", "do", "does",
    "for", "from", "in", "is", "of", "on", "the", "to",
    "was", "were", "what", "where", "which", "who",
    "whose", "with",
}


# Helpers

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the valid 32K BM25 replication with 64-token "
            "and 512-token evidence budgets."
        )
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N questions as a preflight.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"Invalid JSON in {path}, line {line_number}."
                ) from error
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_dataset_row(dataset: Any) -> dict[str, Any]:
    for dataset_row in dataset:
        row = dict(dataset_row)
        metadata = row.get("metadata") or {}
        if metadata.get("source") == SOURCE_NAME:
            return row
    raise RuntimeError(f"Could not find source: {SOURCE_NAME}")


def parse_numbered_statements(context: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"(?<!\d)(\d+)\.\s+(.*?)(?=\s+\d+\.\s+|\Z)",
        flags=re.DOTALL,
    )

    statements: list[dict[str, Any]] = []
    for match in pattern.finditer(context):
        statements.append(
            {
                "number": int(match.group(1)),
                "text": " ".join(
                    match.group(2).strip().replace("\n", " ").split()
                ),
            }
        )

    if not statements:
        raise RuntimeError("No numbered statements were parsed.")
    return statements


def lexical_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\w+", text.casefold(), flags=re.UNICODE)
        if token not in STOPWORDS and len(token) > 1
    ]


def normalize(text: str) -> str:
    text = text.casefold().strip()
    text = re.sub(r"[^\w\s-]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def contains_value(text: str, values: list[str]) -> bool:
    normalized_text = normalize(text)
    return any(
        normalize(value) in normalized_text
        for value in values
        if normalize(value)
    )


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def system_prompt() -> str:
    return (
        "You are answering a synthetic memory benchmark. "
        "Use only the supplied numbered memory statements as "
        "the source of truth, even when they contradict ordinary "
        "world knowledge. The statements are chronological: "
        "higher statement numbers are newer and replace lower "
        "conflicting statements about the same fact. "
        "Return only the shortest possible answer phrase. "
        "Do not explain."
    )


def user_prompt(question: str, evidence: str) -> str:
    return (
        "MEMORY EVIDENCE\n"
        "===============\n"
        f"{evidence}\n\n"
        "QUESTION\n"
        "========\n"
        f"{question}\n\n"
        "ANSWER"
    )


def format_evidence(statements: list[dict[str, Any]]) -> str:
    chronological = sorted(
        statements,
        key=lambda item: int(item["number"]),
    )
    return "\n".join(
        f"{item['number']}. {item['text']}"
        for item in chronological
    )


def evidence_token_count(tokenizer: Any, evidence: str) -> int:
    return len(
        tokenizer.encode(
            evidence,
            add_special_tokens=False,
        )
    )


# Update-chain audit

def literal_occurrence(
    text: str,
    values: list[str],
) -> tuple[int, int, str] | None:
    for value in sorted(values, key=len, reverse=True):
        if not value:
            continue
        match = re.search(
            re.escape(value),
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.start(), match.end(), value
    return None


def chain_from_candidate(
    candidate: dict[str, Any],
    gold_answers: list[str],
    statements: list[dict[str, Any]],
) -> dict[str, Any] | None:
    occurrence = literal_occurrence(
        str(candidate["text"]),
        gold_answers,
    )
    if occurrence is None:
        return None

    start, end, matched_gold = occurrence
    candidate_text = str(candidate["text"])
    prefix = candidate_text[:start]
    suffix = candidate_text[end:]

    chain: list[dict[str, Any]] = []

    for statement in statements:
        text = str(statement["text"])
        folded = text.casefold()

        if not folded.startswith(prefix.casefold()):
            continue
        if suffix and not folded.endswith(suffix.casefold()):
            continue

        value_start = len(prefix)
        value_end = len(text) - len(suffix) if suffix else len(text)

        if value_end < value_start:
            continue

        value = text[value_start:value_end].strip()
        if not value:
            continue

        chain.append(
            {
                "number": int(statement["number"]),
                "text": text,
                "value": value,
            }
        )

    chain.sort(key=lambda item: int(item["number"]))
    if not chain:
        return None

    latest = chain[-1]
    if not contains_value(str(latest["value"]), gold_answers):
        return None

    return {
        "relation_template": f"{prefix}[VALUE]{suffix}",
        "matched_gold_alias": matched_gold,
        "chain": chain,
        "oracle_number": int(latest["number"]),
        "oracle_text": str(latest["text"]),
        "oracle_value": str(latest["value"]),
        "previous_conflicting_statements": chain[:-1],
        "previous_conflict_count": max(0, len(chain) - 1),
    }


def build_update_chains(
    questions: list[str],
    answers: list[list[str]],
    statements: list[dict[str, Any]],
    bm25: BM25Okapi,
) -> list[dict[str, Any]]:
    chains: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []

    for question_index, (question, gold_answers) in enumerate(
        zip(questions, answers, strict=True)
    ):
        scores = bm25.get_scores(lexical_tokens(question))

        candidates: list[tuple[float, int, dict[str, Any]]] = []
        for statement_index, statement in enumerate(statements):
            if literal_occurrence(
                str(statement["text"]),
                list(gold_answers),
            ) is None:
                continue
            candidates.append(
                (
                    float(scores[statement_index]),
                    int(statement["number"]),
                    statement,
                )
            )

        candidates.sort(
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )

        resolved: dict[str, Any] | None = None
        for _, _, candidate in candidates:
            possible = chain_from_candidate(
                candidate=candidate,
                gold_answers=list(gold_answers),
                statements=statements,
            )
            if possible is not None:
                resolved = possible
                break

        if resolved is None:
            unresolved.append(
                {
                    "question_index": question_index,
                    "question": question,
                    "gold_answers": list(gold_answers),
                }
            )
            continue

        chains.append(
            {
                "question_index": question_index,
                "question": question,
                "gold_answers": list(gold_answers),
                **resolved,
            }
        )

    if unresolved:
        raise RuntimeError(
            f"Update-chain audit resolved only "
            f"{len(chains)}/{len(questions)} questions. "
            "No model calls were made."
        )

    return sorted(
        chains,
        key=lambda item: int(item["question_index"]),
    )


def save_chain_audit(
    chains: list[dict[str, Any]],
) -> dict[str, Any]:
    write_jsonl(CHAIN_PATH, chains)

    conflict_count = sum(
        int(chain["previous_conflict_count"] > 0)
        for chain in chains
    )
    chain_sizes = [len(chain["chain"]) for chain in chains]

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "chain_version": CHAIN_VERSION,
        "question_count": len(chains),
        "chains_resolved": len(chains),
        "oracle_is_latest_count": len(chains),
        "conflict_question_count": conflict_count,
        "no_conflict_question_count": len(chains) - conflict_count,
        "minimum_chain_size": min(chain_sizes),
        "maximum_chain_size": max(chain_sizes),
        "mean_chain_size": mean(chain_sizes),
        "chain_file": str(CHAIN_PATH),
    }

    CHAIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAIN_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


# BM25 retrieval and model call

def select_bm25_evidence(
    question: str,
    statements: list[dict[str, Any]],
    bm25: BM25Okapi,
    tokenizer: Any,
    token_budget: int,
) -> list[dict[str, Any]]:
    scores = bm25.get_scores(lexical_tokens(question))

    ranked_indices = sorted(
        range(len(statements)),
        key=lambda index: (
            float(scores[index]),
            -int(statements[index]["number"]),
        ),
        reverse=True,
    )

    selected: list[dict[str, Any]] = []
    used_tokens = 0

    for index in ranked_indices:
        statement = statements[index]
        line = f"{statement['number']}. {statement['text']}"
        line_tokens = len(
            tokenizer.encode(
                line,
                add_special_tokens=False,
            )
        )

        if used_tokens + line_tokens > token_budget:
            continue

        selected.append(
            {
                "number": int(statement["number"]),
                "text": str(statement["text"]),
                "bm25_score": float(scores[index]),
                "line_token_count": line_tokens,
            }
        )
        used_tokens += line_tokens

        if used_tokens == token_budget:
            break

    if not selected:
        raise RuntimeError(
            f"BM25 selected no statements for budget {token_budget}."
        )

    return sorted(
        selected,
        key=lambda item: int(item["number"]),
    )


def ask_model(
    session: requests.Session,
    question: str,
    evidence: str,
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": user_prompt(question, evidence),
            },
        ],
        "stream": False,
        "options": {
            "num_ctx": NUM_CTX,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
            started = time.perf_counter()
            response = session.post(
                f"{BASE_URL}/api/chat",
                json=payload,
                timeout=600,
            )
            elapsed = time.perf_counter() - started

            response.raise_for_status()
            data = response.json()

            content = str(
                (data.get("message") or {}).get("content", "")
            ).strip()
            if not content:
                raise RuntimeError("The model returned an empty answer.")

            return {
                "model_answer": content,
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "done_reason": data.get("done_reason"),
                "client_elapsed_seconds": elapsed,
            }

        except (
            requests.RequestException,
            ValueError,
            RuntimeError,
        ) as error:
            last_error = error

            if attempt == 3:
                break

            wait_seconds = 5 * attempt
            print(
                f"Request failed on attempt {attempt}; "
                f"retrying in {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    raise RuntimeError(
        "Ollama request failed after three attempts."
    ) from last_error


# Scoring and summary

def classify_answer(
    prediction: str,
    gold_answers: list[str],
    stale_values: list[str],
) -> str:
    if contains_value(prediction, gold_answers):
        return "latest"
    if contains_value(prediction, stale_values):
        return "stale"
    if normalize(prediction).startswith("unknown"):
        return "unknown"
    return "other"


def exact_mcnemar_p_value(
    first_only_correct: int,
    second_only_correct: int,
) -> float:
    discordant = first_only_correct + second_only_correct
    if discordant == 0:
        return 1.0

    smaller = min(first_only_correct, second_only_correct)
    lower_tail = sum(
        comb(discordant, value)
        for value in range(smaller + 1)
    ) / (2 ** discordant)

    return min(1.0, 2 * lower_tail)


def paired_comparison(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    first = {
        int(row["question_index"]): row
        for row in rows
        if row["condition"] == "bm25_64"
    }
    second = {
        int(row["question_index"]): row
        for row in rows
        if row["condition"] == "bm25_512"
    }

    shared = sorted(set(first) & set(second))

    both_correct = 0
    first_only = 0
    second_only = 0
    both_wrong = 0
    correct_to_stale = 0

    for index in shared:
        first_correct = bool(first[index]["substring_exact_match"])
        second_correct = bool(second[index]["substring_exact_match"])

        if first_correct and second_correct:
            both_correct += 1
        elif first_correct:
            first_only += 1
            if second[index]["answer_type"] == "stale":
                correct_to_stale += 1
        elif second_correct:
            second_only += 1
        else:
            both_wrong += 1

    return {
        "first_condition": "bm25_64",
        "second_condition": "bm25_512",
        "question_count": len(shared),
        "both_correct": both_correct,
        "first_only_correct": first_only,
        "second_only_correct": second_only,
        "both_wrong": both_wrong,
        "correct_to_stale_count": correct_to_stale,
        "exact_mcnemar_p_value": exact_mcnemar_p_value(
            first_only,
            second_only,
        ),
    }


def build_summary(
    rows: list[dict[str, Any]],
    chain_summary: dict[str, Any],
    run_question_count: int,
) -> dict[str, Any]:
    condition_summaries: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        condition_rows = [
            row for row in rows if row["condition"] == condition
        ]

        for group in ["all", "conflict", "no_conflict"]:
            if group == "all":
                group_rows = condition_rows
            elif group == "conflict":
                group_rows = [
                    row
                    for row in condition_rows
                    if row["has_previous_conflict"]
                ]
            else:
                group_rows = [
                    row
                    for row in condition_rows
                    if not row["has_previous_conflict"]
                ]

            correct_count = sum(
                int(row["substring_exact_match"])
                for row in group_rows
            )
            support_count = sum(
                int(row["oracle_support_retrieved"])
                for row in group_rows
            )
            answer_types = Counter(
                str(row["answer_type"])
                for row in group_rows
            )

            condition_summaries.append(
                {
                    "condition": condition,
                    "group": group,
                    "question_count": len(group_rows),
                    "correct_count": correct_count,
                    "accuracy_percent": (
                        100 * correct_count / len(group_rows)
                        if group_rows
                        else None
                    ),
                    "oracle_support_retrieved_count": support_count,
                    "oracle_support_recall_percent": (
                        100 * support_count / len(group_rows)
                        if group_rows
                        else None
                    ),
                    "latest_count": answer_types["latest"],
                    "stale_count": answer_types["stale"],
                    "unknown_count": answer_types["unknown"],
                    "other_count": answer_types["other"],
                    "mean_evidence_token_count": safe_mean(
                        [
                            float(row["evidence_token_count"])
                            for row in group_rows
                        ]
                    ),
                    "mean_statement_count": safe_mean(
                        [
                            float(row["statement_count"])
                            for row in group_rows
                        ]
                    ),
                    "mean_prompt_eval_count": safe_mean(
                        [
                            float(row["prompt_eval_count"])
                            for row in group_rows
                            if row.get("prompt_eval_count") is not None
                        ]
                    ),
                    "mean_client_elapsed_seconds": safe_mean(
                        [
                            float(row["client_elapsed_seconds"])
                            for row in group_rows
                        ]
                    ),
                }
            )

    return {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "model": MODEL,
        "tokenizer": TOKENIZER_NAME,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "chain_version": CHAIN_VERSION,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "num_ctx": NUM_CTX,
        "run_question_count": run_question_count,
        "expected_call_count": run_question_count * len(CONDITIONS),
        "completed_call_count": len(rows),
        "chain_audit": chain_summary,
        "condition_summaries": condition_summaries,
        "paired_comparison": paired_comparison(rows),
        "support_recall_definition": (
            "Whether the reconstructed newest/oracle statement "
            "was included in the evidence supplied to the model."
        ),
        "scope_note": (
            "The full 32K history is intentionally omitted because "
            "it tokenizes to approximately 37.5K evidence tokens, "
            "exceeding the server model's reported 32,768-token "
            "context limit before prompt overhead."
        ),
        "statistical_warning": (
            "McNemar treats questions as paired benchmark units. "
            "Because all questions share one benchmark history, "
            "the p-value is diagnostic rather than population-level."
        ),
        "result_file": str(OUTPUT_PATH),
        "chain_file": str(CHAIN_PATH),
    }


# Main

def main() -> None:
    args = parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be a positive integer.")

    print("Loading dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
    )
    row = find_dataset_row(dataset)

    questions = [str(item) for item in row["questions"]]
    answers = [
        [str(answer) for answer in answer_list]
        for answer_list in row["answers"]
    ]
    statements = parse_numbered_statements(str(row["context"]))

    print(f"Source: {SOURCE_NAME}")
    print(f"Parsed statements: {len(statements)}")
    print(f"Questions: {len(questions)}")
    print(f"Loading tokenizer: {TOKENIZER_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    tokenized_corpus = [
        lexical_tokens(str(statement["text"]))
        for statement in statements
    ]
    bm25 = BM25Okapi(tokenized_corpus)

    print("Auditing update chains...")
    chains = build_update_chains(
        questions=questions,
        answers=answers,
        statements=statements,
        bm25=bm25,
    )
    chain_summary = save_chain_audit(chains)

    print(
        f"Update chains: {chain_summary['chains_resolved']}/"
        f"{chain_summary['question_count']} resolved"
    )
    print(
        f"Conflict questions: "
        f"{chain_summary['conflict_question_count']}"
    )
    print(
        f"No-conflict questions: "
        f"{chain_summary['no_conflict_question_count']}"
    )

    chain_by_index = {
        int(chain["question_index"]): chain
        for chain in chains
    }

    run_indices = list(range(len(questions)))
    if args.limit is not None:
        run_indices = run_indices[:args.limit]

    existing_rows = load_jsonl(OUTPUT_PATH)
    completed: dict[tuple[int, str], dict[str, Any]] = {}

    for existing in existing_rows:
        if (
            existing.get("source") != SOURCE_NAME
            or existing.get("model") != MODEL
            or existing.get("prompt_version") != PROMPT_VERSION
            or existing.get("retrieval_version") != RETRIEVAL_VERSION
        ):
            raise RuntimeError(
                f"{OUTPUT_PATH} contains incompatible results. "
                "Rename or delete it before running."
            )

        key = (
            int(existing["question_index"]),
            str(existing["condition"]),
        )
        if key in completed:
            raise RuntimeError(
                f"Duplicate result in {OUTPUT_PATH}: {key}"
            )
        completed[key] = existing

    expected_calls = len(run_indices) * len(CONDITIONS)
    relevant_completed = {
        key: value
        for key, value in completed.items()
        if key[0] in set(run_indices)
    }

    pending_keys = [
        (question_index, condition)
        for question_index in run_indices
        for condition in CONDITIONS
        if (question_index, condition) not in completed
    ]

    print(f"Model: {MODEL}")
    print(f"Prompt: {PROMPT_VERSION}")
    print("Conditions: bm25_64, bm25_512")
    print(f"Run questions: {len(run_indices)}")
    print(f"Expected calls for this run: {expected_calls}")
    print(
        f"Already completed in this run: "
        f"{len(relevant_completed)}"
    )
    print(f"Calls remaining: {len(pending_keys)}")

    if pending_keys:
        session = requests.Session()

        if USERNAME:
            password = getpass(
                "\nOllama password: "
            )
            session.auth = (USERNAME, password)
    else:
        session = None
        print(
            "All requested calls already exist; "
            "rebuilding summary."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if OUTPUT_PATH.exists() else "w"

    with OUTPUT_PATH.open(
        output_mode,
        encoding="utf-8",
    ) as output_file:
        call_position = len(relevant_completed)

        for question_index in run_indices:
            question = questions[question_index]
            gold_answers = answers[question_index]
            chain = chain_by_index[question_index]

            oracle_number = int(chain["oracle_number"])
            stale_values = [
                str(item["value"])
                for item in chain[
                    "previous_conflicting_statements"
                ]
            ]
            has_previous_conflict = bool(
                chain["previous_conflict_count"] > 0
            )

            evidence_by_condition = {
                "bm25_64": select_bm25_evidence(
                    question=question,
                    statements=statements,
                    bm25=bm25,
                    tokenizer=tokenizer,
                    token_budget=64,
                ),
                "bm25_512": select_bm25_evidence(
                    question=question,
                    statements=statements,
                    bm25=bm25,
                    tokenizer=tokenizer,
                    token_budget=512,
                ),
            }

            for condition in CONDITIONS:
                key = (question_index, condition)
                if key in completed:
                    continue

                if session is None:
                    raise RuntimeError(
                        "No Ollama session for a pending call."
                    )

                evidence_items = evidence_by_condition[condition]
                evidence = format_evidence(evidence_items)

                call_position += 1
                model_result = ask_model(
                    session=session,
                    question=question,
                    evidence=evidence,
                )

                correct = contains_value(
                    model_result["model_answer"],
                    gold_answers,
                )

                answer_type = classify_answer(
                    prediction=model_result["model_answer"],
                    gold_answers=gold_answers,
                    stale_values=stale_values,
                )

                evidence_numbers = {
                    int(item["number"])
                    for item in evidence_items
                }
                oracle_support_retrieved = (
                    oracle_number in evidence_numbers
                )

                result = {
                    "dataset": DATASET_NAME,
                    "split": SPLIT_NAME,
                    "source": SOURCE_NAME,
                    "question_index": question_index,
                    "question": question,
                    "gold_answers": gold_answers,
                    "condition": condition,
                    "model": MODEL,
                    "prompt_version": PROMPT_VERSION,
                    "retrieval_version": RETRIEVAL_VERSION,
                    "chain_version": CHAIN_VERSION,
                    "requested_num_ctx": NUM_CTX,
                    "temperature": TEMPERATURE,
                    "seed": SEED,
                    "num_predict": NUM_PREDICT,
                    "has_previous_conflict":
                        has_previous_conflict,
                    "oracle_number": oracle_number,
                    "oracle_value": chain["oracle_value"],
                    "stale_values": stale_values,
                    "oracle_support_retrieved":
                        oracle_support_retrieved,
                    "statement_count": len(evidence_items),
                    "evidence_token_count":
                        evidence_token_count(tokenizer, evidence),
                    "evidence_statement_numbers": [
                        int(item["number"])
                        for item in evidence_items
                    ],
                    "evidence_statements": evidence_items,
                    "substring_exact_match": correct,
                    "answer_type": answer_type,
                    **model_result,
                }

                completed[key] = result
                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    ) + "\n"
                )
                output_file.flush()

                print(
                    f"{call_position:3}/{expected_calls} "
                    f"| q={question_index + 1:3} "
                    f"| {condition:<9} "
                    f"| "
                    f"{'CORRECT' if correct else 'WRONG':<7} "
                    f"| {answer_type:<7} "
                    f"| support="
                    f"{'yes' if oracle_support_retrieved else 'NO'}"
                )

    final_rows = [
        completed[(question_index, condition)]
        for question_index in run_indices
        for condition in CONDITIONS
        if (question_index, condition) in completed
    ]

    if len(final_rows) != expected_calls:
        raise RuntimeError(
            f"Expected {expected_calls} completed calls, "
            f"found {len(final_rows)}."
        )

    summary = build_summary(
        rows=final_rows,
        chain_summary=chain_summary,
        run_question_count=len(run_indices),
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 108)
    print("32K BM25 REPLICATION SUMMARY")
    print("=" * 108)

    for item in summary["condition_summaries"]:
        if item["group"] != "all":
            continue

        print(
            f"{item['condition']:<9} "
            f"{item['correct_count']:>3}/"
            f"{item['question_count']:<3} "
            f"({item['accuracy_percent']:>5.1f}%) "
            f"| support="
            f"{item['oracle_support_retrieved_count']:>3}/"
            f"{item['question_count']:<3} "
            f"({item['oracle_support_recall_percent']:>5.1f}%) "
            f"| latest={item['latest_count']:>3} "
            f"| stale={item['stale_count']:>3} "
            f"| unknown={item['unknown_count']:>3} "
            f"| other={item['other_count']:>3}"
        )

    print()
    print("CONFLICT STRATIFICATION")
    for item in summary["condition_summaries"]:
        if item["group"] == "all":
            continue

        accuracy = (
            f"{item['accuracy_percent']:.1f}%"
            if item["accuracy_percent"] is not None
            else "n/a"
        )

        print(
            f"{item['condition']:<9} "
            f"| {item['group']:<11} "
            f"| {item['correct_count']:>3}/"
            f"{item['question_count']:<3} "
            f"({accuracy})"
        )

    paired = summary["paired_comparison"]
    print()
    print("PAIRED BM25-64 VS BM25-512")
    print(
        f"64-only correct: "
        f"{paired['first_only_correct']}"
    )
    print(
        f"512-only correct: "
        f"{paired['second_only_correct']}"
    )
    print(
        f"Correct-to-stale flips: "
        f"{paired['correct_to_stale_count']}"
    )
    print(
        f"Exact McNemar p-value: "
        f"{paired['exact_mcnemar_p_value']:.8f}"
    )

    print()
    print(f"Detailed results: {OUTPUT_PATH}")
    print(f"Summary:          {SUMMARY_PATH}")
    print(f"Update chains:    {CHAIN_PATH}")
    print(f"Chain summary:    {CHAIN_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
