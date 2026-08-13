from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DATASET_NAME = "ai-hyz/MemoryAgentBench"
SPLIT_NAME = "Conflict_Resolution"
SOURCE_NAME = "factconsolidation_sh_6k"

ORACLE_AUDIT_PATH = Path(
    "results/09_bm25_retrieval_audit.jsonl"
)

OUTPUT_PATH = Path(
    "results/14_update_chains.jsonl"
)

SUMMARY_PATH = Path(
    "results/14_update_chains_summary.json"
)


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
                "text": clean_text(match.group(2)),
            }
        )

    return statements


def clean_text(text: str) -> str:
    return " ".join(
        text.strip().replace("\n", " ").split()
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


def locate_gold_span(
    statement_text: str,
    gold_answers: list[str],
) -> tuple[str, str, str] | None:
    """
    Find a gold answer inside the oracle statement.

    Returns:
        matched_gold, prefix_before_answer, suffix_after_answer
    """
    for gold in sorted(
        gold_answers,
        key=len,
        reverse=True,
    ):
        match = re.search(
            re.escape(gold),
            statement_text,
            flags=re.IGNORECASE,
        )

        if match:
            return (
                gold,
                statement_text[:match.start()],
                statement_text[match.end():],
            )

    return None


def build_chain_pattern(
    prefix: str,
    suffix: str,
) -> re.Pattern[str]:
    prefix = clean_text(prefix)
    suffix = clean_text(suffix)

    if suffix:
        expression = (
            r"^"
            + re.escape(prefix)
            + r"\s*(?P<value>.+?)\s*"
            + re.escape(suffix)
            + r"$"
        )
    else:
        expression = (
            r"^"
            + re.escape(prefix)
            + r"\s*(?P<value>.+?)\s*$"
        )

    return re.compile(
        expression,
        flags=re.IGNORECASE,
    )


def extract_chain(
    statements: list[dict[str, Any]],
    oracle_statement: dict[str, Any],
    gold_answers: list[str],
) -> dict[str, Any]:
    oracle_text = clean_text(
        str(oracle_statement["text"])
    )

    located = locate_gold_span(
        statement_text=oracle_text,
        gold_answers=gold_answers,
    )

    if located is None:
        return {
            "resolved": False,
            "reason": (
                "Gold answer was not found in the "
                "oracle statement text."
            ),
        }

    matched_gold, prefix, suffix = located

    pattern = build_chain_pattern(
        prefix=prefix,
        suffix=suffix,
    )

    chain: list[dict[str, Any]] = []

    for statement in statements:
        match = pattern.match(
            clean_text(statement["text"])
        )

        if not match:
            continue

        chain.append(
            {
                "number": int(statement["number"]),
                "text": statement["text"],
                "value": clean_text(
                    match.group("value")
                ),
            }
        )

    chain.sort(
        key=lambda item: item["number"]
    )

    oracle_number = int(
        oracle_statement["number"]
    )

    previous = [
        item
        for item in chain
        if item["number"] < oracle_number
    ]

    previous_conflicts = [
        item
        for item in previous
        if normalize(item["value"])
        != normalize(matched_gold)
    ]

    later = [
        item
        for item in chain
        if item["number"] > oracle_number
    ]

    return {
        "resolved": True,
        "matched_gold": matched_gold,
        "relation_prefix": clean_text(prefix),
        "relation_suffix": clean_text(suffix),
        "oracle_number": oracle_number,
        "oracle_text": oracle_text,
        "chain": chain,
        "chain_size": len(chain),
        "previous_statements": previous,
        "previous_conflicting_statements":
            previous_conflicts,
        "previous_conflict_count":
            len(previous_conflicts),
        "later_statements": later,
        "oracle_is_latest": len(later) == 0,
    }


def main() -> None:
    if not ORACLE_AUDIT_PATH.exists():
        raise FileNotFoundError(
            f"Missing file: {ORACLE_AUDIT_PATH}"
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

    audit_rows = load_jsonl(
        ORACLE_AUDIT_PATH
    )

    print(
        f"Parsed {len(statements)} statements."
    )
    print(
        f"Loaded {len(audit_rows)} questions."
    )

    output_rows: list[dict[str, Any]] = []
    chain_sizes: list[int] = []
    conflict_counts: list[int] = []

    resolved_count = 0
    oracle_latest_count = 0
    with_previous_conflict_count = 0

    unresolved_reasons: Counter[str] = Counter()

    for position, audit in enumerate(
        audit_rows,
        start=1,
    ):
        oracle = audit.get(
            "gold_assisted_oracle_statement"
        )

        gold_answers = [
            str(answer)
            for answer in audit["gold_answers"]
        ]

        if not oracle:
            chain_result = {
                "resolved": False,
                "reason": (
                    "No oracle statement was available."
                ),
            }
        else:
            chain_result = extract_chain(
                statements=statements,
                oracle_statement=oracle,
                gold_answers=gold_answers,
            )

        output_row = {
            "question_index":
                int(audit["question_index"]),
            "question": audit["question"],
            "gold_answers": gold_answers,
            **chain_result,
        }

        output_rows.append(output_row)

        print()
        print("=" * 76)
        print(
            f"QUESTION {position}: "
            f"{audit['question']}"
        )

        if not chain_result["resolved"]:
            reason = chain_result["reason"]
            unresolved_reasons[reason] += 1
            print("UNRESOLVED:", reason)
            continue

        resolved_count += 1

        chain_sizes.append(
            chain_result["chain_size"]
        )

        conflict_counts.append(
            chain_result[
                "previous_conflict_count"
            ]
        )

        if chain_result["oracle_is_latest"]:
            oracle_latest_count += 1

        if (
            chain_result[
                "previous_conflict_count"
            ]
            > 0
        ):
            with_previous_conflict_count += 1

        if position <= 20:
            print(
                "Relation:",
                chain_result["relation_prefix"],
                "[VALUE]",
                chain_result["relation_suffix"],
            )

            print("Chain:")

            for item in chain_result["chain"]:
                marker = (
                    "  <-- GOLD/LATEST"
                    if item["number"]
                    == chain_result["oracle_number"]
                    else ""
                )

                print(
                    f"  {item['number']}. "
                    f"{item['text']}"
                    f"{marker}"
                )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        for output_row in output_rows:
            output_file.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    summary = {
        "dataset": DATASET_NAME,
        "split": SPLIT_NAME,
        "source": SOURCE_NAME,
        "question_count": len(output_rows),
        "resolved_count": resolved_count,
        "unresolved_count": (
            len(output_rows) - resolved_count
        ),
        "oracle_is_latest_count":
            oracle_latest_count,
        "questions_with_previous_conflict":
            with_previous_conflict_count,
        "minimum_chain_size": (
            min(chain_sizes)
            if chain_sizes
            else None
        ),
        "maximum_chain_size": (
            max(chain_sizes)
            if chain_sizes
            else None
        ),
        "mean_chain_size": (
            statistics.mean(chain_sizes)
            if chain_sizes
            else None
        ),
        "minimum_previous_conflicts": (
            min(conflict_counts)
            if conflict_counts
            else None
        ),
        "maximum_previous_conflicts": (
            max(conflict_counts)
            if conflict_counts
            else None
        ),
        "mean_previous_conflicts": (
            statistics.mean(conflict_counts)
            if conflict_counts
            else None
        ),
        "unresolved_reasons":
            dict(unresolved_reasons),
        "chain_file": str(OUTPUT_PATH),
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
    print("=" * 76)
    print("UPDATE-CHAIN AUDIT SUMMARY")
    print(
        "Chains resolved:",
        f"{resolved_count}/{len(output_rows)}",
    )
    print(
        "Oracle is latest:",
        f"{oracle_latest_count}/{resolved_count}",
    )
    print(
        "Questions with an older conflicting fact:",
        f"{with_previous_conflict_count}/"
        f"{resolved_count}",
    )
    print(
        "Chain-size range:",
        summary["minimum_chain_size"],
        "to",
        summary["maximum_chain_size"],
    )
    print(
        "Mean chain size:",
        (
            f"{summary['mean_chain_size']:.2f}"
            if summary["mean_chain_size"]
            is not None
            else "N/A"
        ),
    )
    print(
        "Mean previous conflicts:",
        (
            f"{summary['mean_previous_conflicts']:.2f}"
            if summary[
                "mean_previous_conflicts"
            ] is not None
            else "N/A"
        ),
    )

    print(f"\nDetailed chains: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()