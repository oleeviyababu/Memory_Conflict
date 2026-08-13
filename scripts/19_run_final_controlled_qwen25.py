from __future__ import annotations

import hashlib
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


# Experiment configuration


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

CHAIN_PATH = Path("results/14_update_chains.jsonl")

OUTPUT_PATH = Path(
    "results/19_final_controlled_qwen25.jsonl"
)
SUMMARY_PATH = Path(
    "results/19_final_controlled_qwen25_summary.json"
)

BASE_URL = os.environ.get(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
).rstrip("/")

USERNAME = os.environ.get("OLLAMA_USERNAME")
MODEL = "qwen2.5:7b-instruct"

NUM_CTX = 2048
TEMPERATURE = 0
SEED = 42
NUM_PREDICT = 64

PROMPT_VERSION = "frozen_non_abstaining_v1"

CONDITION_A = "latest_plus_3_unrelated"
CONDITION_B = "old_plus_latest_plus_2_unrelated"
CONDITION_ORDER = [CONDITION_A, CONDITION_B]

EXPECTED_QUESTIONS = 74
EXPECTED_CALLS = EXPECTED_QUESTIONS * len(CONDITION_ORDER)

STOPWORDS = {
    "a", "an", "and", "at", "by", "did", "do", "does",
    "for", "from", "in", "is", "of", "on", "the", "to",
    "was", "were", "what", "where", "which", "who",
    "whose", "with",
}


# File and dataset helpers


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
                    f"Invalid JSON in {path}, line {line_number}"
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



# Text, scoring, and distractor helpers


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


def substring_exact_match(
    prediction: str,
    gold_answers: list[str],
) -> bool:
    return contains_value(
        prediction=prediction,
        values=gold_answers,
    )


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

        # Unrelated distractors must share no content word
        # with the question.
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
            f"Could select only {len(selected)} unrelated "
            f"distractors for question {question_index}; "
            f"required {count}."
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


# Ollama request

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
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_predict": NUM_PREDICT,
        },
    }

    last_error: Exception | None = None

    for attempt in range(1, 4):
        try:
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
                "eval_count":
                    data.get("eval_count"),
                "client_elapsed_seconds":
                    elapsed,
            }

        except (
            requests.RequestException,
            ValueError,
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


# Statistics and summary

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


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def build_summary(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    by_condition: dict[
        str,
        dict[int, dict[str, Any]],
    ] = {
        condition: {}
        for condition in CONDITION_ORDER
    }

    for result in results:
        condition = str(result["condition"])
        question_index = int(
            result["question_index"]
        )

        by_condition[condition][question_index] = (
            result
        )

    condition_summaries: list[dict[str, Any]] = []

    for condition in CONDITION_ORDER:
        rows = list(
            by_condition[condition].values()
        )

        correct_count = sum(
            int(row["substring_exact_match"])
            for row in rows
        )

        answer_types = Counter(
            str(row["answer_type"])
            for row in rows
        )

        latencies = [
            float(row["client_elapsed_seconds"])
            for row in rows
            if row.get("client_elapsed_seconds")
            is not None
        ]

        prompt_counts = [
            float(row["prompt_eval_count"])
            for row in rows
            if row.get("prompt_eval_count")
            is not None
        ]

        condition_summaries.append(
            {
                "condition": condition,
                "question_count": len(rows),
                "correct_count": correct_count,
                "accuracy_percent": (
                    100 * correct_count / len(rows)
                    if rows
                    else None
                ),
                "latest_count":
                    answer_types["latest"],
                "stale_count":
                    answer_types["stale"],
                "unknown_count":
                    answer_types["unknown"],
                "other_count":
                    answer_types["other"],
                "mean_prompt_eval_count":
                    safe_mean(prompt_counts),
                "mean_client_elapsed_seconds":
                    safe_mean(latencies),
            }
        )

    first = by_condition[CONDITION_A]
    second = by_condition[CONDITION_B]

    shared_indices = sorted(
        set(first) & set(second)
    )

    both_correct = 0
    first_only_correct = 0
    second_only_correct = 0
    both_wrong = 0

    correct_to_stale = 0
    flip_examples: list[dict[str, Any]] = []

    for index in shared_indices:
        first_row = first[index]
        second_row = second[index]

        first_correct = bool(
            first_row["substring_exact_match"]
        )
        second_correct = bool(
            second_row["substring_exact_match"]
        )

        if first_correct and second_correct:
            both_correct += 1
        elif first_correct and not second_correct:
            first_only_correct += 1

            if (
                second_row["answer_type"]
                == "stale"
            ):
                correct_to_stale += 1

            if len(flip_examples) < 15:
                flip_examples.append(
                    {
                        "question_index": index,
                        "question":
                            first_row["question"],
                        "gold_answers":
                            first_row["gold_answers"],
                        "stale_values":
                            second_row["stale_values"],
                        "condition_a_answer":
                            first_row["model_answer"],
                        "condition_b_answer":
                            second_row["model_answer"],
                        "condition_b_answer_type":
                            second_row["answer_type"],
                    }
                )
        elif second_correct:
            second_only_correct += 1
        else:
            both_wrong += 1

    paired = {
        "first_condition": CONDITION_A,
        "second_condition": CONDITION_B,
        "question_count": len(shared_indices),
        "both_correct": both_correct,
        "first_only_correct":
            first_only_correct,
        "second_only_correct":
            second_only_correct,
        "both_wrong": both_wrong,
        "accuracy_difference_points": (
            100
            * (
                first_only_correct
                - second_only_correct
            )
            / len(shared_indices)
            if shared_indices
            else None
        ),
        "correct_to_stale_count":
            correct_to_stale,
        "exact_mcnemar_p_value":
            exact_mcnemar_p_value(
                first_only_correct,
                second_only_correct,
            ),
        "flip_examples": flip_examples,
    }

    return {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "requested_num_ctx": NUM_CTX,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "num_predict": NUM_PREDICT,
        "expected_question_count":
            EXPECTED_QUESTIONS,
        "expected_call_count":
            EXPECTED_CALLS,
        "completed_call_count":
            len(results),
        "conditions": condition_summaries,
        "paired_comparison": paired,
        "distractor_selection": (
            "Deterministic statements with zero "
            "content-word overlap with the question, "
            "excluding update-chain values."
        ),
        "statistical_warning": (
            "McNemar treats benchmark questions as paired "
            "units. These questions share one benchmark "
            "history, so the p-value is diagnostic rather "
            "than population-level evidence."
        ),
        "result_file": str(OUTPUT_PATH),
    }


# Main experiment

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

    all_chain_rows = sorted(
        load_jsonl(CHAIN_PATH),
        key=lambda item: int(
            item["question_index"]
        ),
    )

    chain_rows = [
        row
        for row in all_chain_rows
        if int(
            row["previous_conflict_count"]
        ) > 0
    ]

    if len(chain_rows) != EXPECTED_QUESTIONS:
        raise RuntimeError(
            f"Expected {EXPECTED_QUESTIONS} conflict "
            f"questions, found {len(chain_rows)}."
        )

    print(
        f"Parsed {len(statements)} statements."
    )
    print(
        f"Conflict questions: {len(chain_rows)}"
    )
    print(
        f"Conditions per question: "
        f"{len(CONDITION_ORDER)}"
    )
    print(
        f"Planned calls: {EXPECTED_CALLS}"
    )
    print(f"Model: {MODEL}")
    print(
        f"Prompt: {PROMPT_VERSION}"
    )
    print(
        "Conditions:"
        f"\n  A. {CONDITION_A}"
        f"\n  B. {CONDITION_B}"
    )

    existing_rows = load_jsonl(OUTPUT_PATH)

    completed: dict[
        tuple[int, str],
        dict[str, Any],
    ] = {}

    for existing in existing_rows:
        if (
            existing.get("prompt_version")
            != PROMPT_VERSION
        ):
            raise RuntimeError(
                f"{OUTPUT_PATH} contains results from "
                "a different prompt version. Rename or "
                "delete the file before running."
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

    if completed:
        print(
            f"Resume mode: {len(completed)} completed "
            "calls found."
        )

    pending_calls = (
        EXPECTED_CALLS - len(completed)
    )

    if pending_calls < 0:
        raise RuntimeError(
            "Output file contains more rows than expected."
        )

    if pending_calls == 0:
        print(
            "All calls are already complete. "
            "Rebuilding summary only."
        )
        session = None
    else:
        print(
            f"Calls remaining: {pending_calls}"
        )

        session = requests.Session()

        if USERNAME:
            password = getpass(
                "\nOllama password: "
            )
            session.auth = (USERNAME, password)
            
            

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_mode = (
        "a"
        if OUTPUT_PATH.exists()
        else "w"
    )

    with OUTPUT_PATH.open(
        output_mode,
        encoding="utf-8",
    ) as output_file:
        call_position = len(completed)

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

            if not previous_conflicts:
                raise RuntimeError(
                    f"Conflict question {question_index} "
                    "has no previous conflicting statement."
                )

            conflict = dict(
                previous_conflicts[-1]
            )

            stale_values = [
                str(item["value"])
                for item in previous_conflicts
            ]

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
                tuple[
                    str,
                    list[dict[str, Any]],
                ]
            ] = [
                (
                    CONDITION_A,
                    [latest] + distractors,
                ),
                (
                    CONDITION_B,
                    [
                        conflict,
                        latest,
                        distractors[0],
                        distractors[1],
                    ],
                ),
            ]

            for condition, evidence_items in conditions:
                key = (
                    question_index,
                    condition,
                )

                if key in completed:
                    continue

                if session is None:
                    raise RuntimeError(
                        "No session available for pending call."
                    )

                call_position += 1

                evidence = format_evidence(
                    evidence_items
                )

                model_result = ask_model(
                    session=session,
                    question=question,
                    evidence=evidence,
                )

                answer_type = classify_answer(
                    prediction=model_result[
                        "model_answer"
                    ],
                    gold_answers=gold_answers,
                    stale_values=stale_values,
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
                    "question":
                        question,
                    "gold_answers":
                        gold_answers,
                    "stale_values":
                        stale_values,
                    "condition":
                        condition,
                    "statement_count":
                        len(evidence_items),
                    "evidence_statement_numbers":
                        [
                            int(item["number"])
                            for item in sorted(
                                evidence_items,
                                key=lambda item:
                                    int(item["number"]),
                            )
                        ],
                    "evidence_statements":
                        sorted(
                            evidence_items,
                            key=lambda item:
                                int(item["number"]),
                        ),
                    "model":
                        MODEL,
                    "prompt_version":
                        PROMPT_VERSION,
                    "requested_num_ctx":
                        NUM_CTX,
                    "temperature":
                        TEMPERATURE,
                    "seed":
                        SEED,
                    "num_predict":
                        NUM_PREDICT,
                    "substring_exact_match":
                        correct,
                    "answer_type":
                        answer_type,
                    **model_result,
                }

                completed[key] = result

                output_file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output_file.flush()

                print(
                    f"{call_position:3}/{EXPECTED_CALLS} "
                    f"| q={question_index + 1:3} "
                    f"| {condition:<38} "
                    f"| "
                    f"{'CORRECT' if correct else 'WRONG':<7} "
                    f"| {answer_type}"
                )

    final_results = [
        completed[key]
        for key in sorted(
            completed,
            key=lambda item: (
                item[0],
                CONDITION_ORDER.index(
                    item[1]
                ),
            ),
        )
    ]

    if len(final_results) != EXPECTED_CALLS:
        raise RuntimeError(
            f"Expected {EXPECTED_CALLS} completed calls, "
            f"found {len(final_results)}."
        )

    summary = build_summary(final_results)

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("FINAL CONTROLLED QWEN 2.5 SUMMARY")
    print("=" * 104)

    for item in summary["conditions"]:
        print(
            f"{item['condition']:<42} "
            f"{item['correct_count']:>2}/"
            f"{item['question_count']:<2} "
            f"({item['accuracy_percent']:>5.1f}%) "
            f"| latest={item['latest_count']:>2} "
            f"| stale={item['stale_count']:>2} "
            f"| unknown={item['unknown_count']:>2} "
            f"| other={item['other_count']:>2}"
        )

    paired = summary["paired_comparison"]

    print()
    print("PAIRED EQUAL-SIZE COMPARISON")
    print(
        f"A only correct: "
        f"{paired['first_only_correct']}"
    )
    print(
        f"B only correct: "
        f"{paired['second_only_correct']}"
    )
    print(
        f"Correct-to-stale flips: "
        f"{paired['correct_to_stale_count']}"
    )
    print(
        f"Accuracy difference: "
        f"{paired['accuracy_difference_points']:+.1f} "
        "percentage points"
    )
    print(
        f"Exact McNemar p-value: "
        f"{paired['exact_mcnemar_p_value']:.8f}"
    )

    print()
    print(
        f"Detailed results: {OUTPUT_PATH}"
    )
    print(
        f"Summary:          {SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
