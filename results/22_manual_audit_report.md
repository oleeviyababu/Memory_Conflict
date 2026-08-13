# Manual Audit of Memory-Conflict Experiments

## Scope

Twenty cases were manually reviewed across the frozen Qwen 2.5 experiment, the frozen Qwen 3 experiment, the 32K BM25 replication, earlier UNKNOWN-prompt diagnostics, and a semantic-scoring edge case.

## Main findings

1. **Principal controlled flips are genuine.** All 9 sampled 6K model flips (5 Qwen 2.5 and 4 Qwen 3) changed from the newest value to the explicitly supplied obsolete value.
2. **The 32K budget flips are genuine.** All 4 sampled BM25-64 to BM25-512 flips selected the obsolete value while both newest and obsolete target statements were present.
3. **No final frozen-prompt case returned UNKNOWN.** UNKNOWN examples belong only to the earlier abstaining-prompt diagnostic and should be described as prompt sensitivity.
4. **One known metric false negative exists.** `goalie` is semantically equivalent to the gold `goaltender`, although normalized substring match scores it as incorrect.
5. **Two 32K chain-audit ambiguities were found (questions 9 and 10).** The chain builder attached the gold value to an unrelated statement, even though the retrieved evidence contains the true gold statement and another direct contradictory target statement.

## 32K analysis correction

- Direct literal gold-support presence is **100/100 for BM25-64** and **100/100 for BM25-512**.
- Keep the official 100-question aggregate accuracies unchanged.
- Flag questions 9 and 10 as ambiguous for chain-based stratification.
- On the clean no-conflict subset (n=35), accuracy is **34/35 (97.1%)** for BM25-64 and **35/35 (100.0%)** for BM25-512.
- Excluding the two ambiguous cases, the paired comparison is 18 BM25-64-only correct versus 6 BM25-512-only correct (exact McNemar p=0.02266).

## Reviewer-facing interpretation

The manual audit strengthens the central claim: sampled failures in the principal experiments are not vague semantic mistakes; they are direct selections of obsolete values. The audit also identifies a limited chain-construction issue in two 32K questions. This does not require new model calls, but the paper should use direct literal support presence and should not count those two questions as clean no-conflict examples.

## Audited cases

| ID | Category | Model | Q | Gold/latest | Obsolete | Answer A | Answer B | Manual verdict |
|---:|---|---|---:|---|---|---|---|---|
| 1 | Qwen 2.5 correct-to-stale flip | qwen2.5:7b-instruct | 2 | The Fairly OddParents | Rurouni Kenshin | The Fairly OddParents | Rurouni Kenshin | Genuine stale selection |
| 2 | Qwen 2.5 correct-to-stale flip | qwen2.5:7b-instruct | 6 | Thomas Kyd | Pierre Beaumarchais | Thomas Kyd | Pierre Beaumarchais | Genuine stale selection |
| 3 | Qwen 2.5 correct-to-stale flip | qwen2.5:7b-instruct | 9 | Swedish | Japanese | Swedish | Japanese | Genuine stale selection |
| 4 | Qwen 2.5 correct-to-stale flip | qwen2.5:7b-instruct | 20 | Steve Jobs | Satya Nadella | Steve Jobs | Satya Nadella | Genuine stale selection |
| 5 | Qwen 2.5 correct-to-stale flip | qwen2.5:7b-instruct | 35 | Taipei | Jerusalem | Taipei | Jerusalem | Genuine stale selection |
| 6 | Qwen 3 correct-to-stale flip | qwen3:14b (think=false) | 44 | Oderzo | Canberra | Oderzo. | Canberra. | Genuine stale selection |
| 7 | Qwen 3 correct-to-stale flip | qwen3:14b (think=false) | 57 | Italy | France | Italy. | France. | Genuine stale selection |
| 8 | Qwen 3 correct-to-stale flip | qwen3:14b (think=false) | 59 | Kylie Minogue | Emma Watson | Kylie Minogue. | Emma Watson. | Genuine stale selection |
| 9 | Qwen 3 correct-to-stale flip | qwen3:14b (think=false) | 92 | Australia | India | Australia. | India. | Genuine stale selection |
| 10 | 32K BM25 correct-to-stale flip | qwen2.5:7b-instruct | 2 | flanker | goalkeeper | flanker | goalkeeper | Genuine stale selection |
| 11 | 32K BM25 correct-to-stale flip | qwen2.5:7b-instruct | 12 | Israel | United Kingdom | Israel | United Kingdom | Genuine stale selection |
| 12 | 32K BM25 correct-to-stale flip | qwen2.5:7b-instruct | 42 | Vijay Mallya | Tim Cook | Vijay Mallya | Tim Cook | Genuine stale selection |
| 13 | 32K BM25 correct-to-stale flip | qwen2.5:7b-instruct | 99 | Mauro Carlesse | Moon Jae-in | Mauro Carlesse | Moon Jae-in | Genuine stale selection |
| 14 | Stable correct latest answer | qwen2.5:7b-instruct | 1 | pesäpallo | ice hockey | pesäpallo | pesäpallo | Correct latest selection |
| 15 | Stable correct latest answer | qwen3:14b (think=false) | 3 | India | England | India. | India. | Correct latest selection |
| 16 | Exploratory UNKNOWN / prompt sensitivity | qwen2.5:7b-instruct | 2 | The Fairly OddParents | Rurouni Kenshin | UNKNOWN |  | Unnecessary abstention |
| 17 | Exploratory UNKNOWN / prompt sensitivity | qwen2.5:7b-instruct | 9 | Swedish | Japanese | UNKNOWN |  | Unnecessary abstention |
| 18 | Semantic scoring artifact | qwen2.5:7b-instruct | 81 | goaltender | center | goalie |  | Semantically correct synonym |
| 19 | 32K chain/benchmark ambiguity | qwen2.5:7b-instruct | 9 | London | Automated chain label was empty | Washington, D.C. | Washington, D.C. | Automated chain label is unreliable for this case |
| 20 | 32K chain/benchmark ambiguity | qwen2.5:7b-instruct | 10 | Rome | Automated chain label was empty | Watertown | Rome | Automated chain label is unreliable for this case |

The accompanying CSV contains the full questions, relevant evidence, official verdicts, and case-specific notes.