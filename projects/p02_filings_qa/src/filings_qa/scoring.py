"""Score generated answers: accuracy, citation of an evidence page, and support by the cited passages.

Three measures, from cheapest to most expensive:

* ``numeric_match`` (free): for questions whose gold answer is one figure,
  whether the answer states that figure. Rounding to the precision the
  answer shows is accepted, as is the same figure written in thousands,
  millions or billions.
* ``cites_evidence`` (free): whether a cited passage lies on an evidence page.
* ``judge`` (paid): a Claude model compares the answer with the gold answer
  (correct, incorrect, or declined) and checks whether the cited passages
  support every figure and claim in the answer.
"""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

from filings_qa.config import QUESTIONS_PATH, RUNS_DIR
from filings_qa.evaluate_retrieval import CONTEXTS_DIR, page_of
from filings_qa.generation import ClaudeGenerator, cost_usd

_NUMBER = re.compile(r"\(?-?\$?\s?\d[\d,]*(?:\.\d+)?\)?")


def parse_numbers(text: str) -> list[tuple[float, int]]:
    """Return ``(value, decimals)`` for every number; ``(1,234)`` is read as a negative figure."""
    out = []
    for match in _NUMBER.finditer(text):
        token = match.group(0)
        negative = token.startswith("(") and token.endswith(")") or "-" in token
        digits = re.sub(r"[^\d.]", "", token)
        if not digits or digits == ".":
            continue
        decimals = len(digits.split(".")[1]) if "." in digits else 0
        try:
            value = float(digits)
        except ValueError:
            continue
        out.append((-value if negative else value, decimals))
    return out


def figure_matches(gold: float, predicted: float, decimals: int, rel_tol: float = 0.01) -> bool:
    """True when ``predicted`` equals ``gold`` within 1 percent or within the rounding of ``predicted``,
    also after a change of scale by a power of one thousand, or by 100 (a ratio written as a percentage)."""
    if gold == 0:
        return predicted == 0
    scales = [1000.0**power for power in range(-3, 4)] + [100.0, 0.01]
    for scale in scales:
        scaled = gold * scale
        rounding = 0.5 * 10.0**-decimals
        if abs(predicted - scaled) <= max(rel_tol * abs(scaled), rounding) and math.copysign(
            1, predicted
        ) == math.copysign(1, scaled):
            return True
    return False


def numeric_match(gold_answer: str, answer: str) -> bool | None:
    """None when the gold answer is not a single figure; otherwise whether the answer states it."""
    gold = parse_numbers(gold_answer)
    if len(gold) != 1:
        return None
    value = gold[0][0]
    return any(figure_matches(value, p, d) for p, d in parse_numbers(answer))


def cites_evidence(cited_chunk_ids: list[str], evidence_pages: list[str]) -> bool:
    return any(page_of(c) in set(evidence_pages) for c in cited_chunk_ids)


JUDGE_PROMPT = """You grade answers to questions about company filings.
You receive the question, the reference answer written by an analyst, the answer to grade, and the
passages the answer cites. Grade two things.

1. correctness:
   - "correct": the answer gives the same result as the reference answer. Small differences in rounding
     or units are acceptable when the figure is the same; extra correct detail is acceptable.
   - "incorrect": the answer gives a different result, misses the main point, or contradicts the reference.
   - "declined": the answer says that the information is not available, without giving a result.
2. supported: true when every figure and factual claim in the answer can be found in, or computed from,
   the cited passages. False when a claim has no basis in the cited passages or no passage is cited.

Return JSON with "correctness", "supported" and a one sentence "reason"."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "string", "enum": ["correct", "incorrect", "declined"]},
        "supported": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "supported", "reason"],
    "additionalProperties": False,
}


def judge_message(question: str, reference: str, answer: str, cited: pd.DataFrame) -> str:
    passages = "\n\n".join(
        f"[{p.position}] {p.company}, {p.doc_type} {p.doc_period}, page {p.page + 1}\n{p.text}"
        for p in cited.itertuples(index=False)
    )
    return (
        f"Question: {question}\n\nReference answer: {reference}\n\nAnswer to grade: {answer}\n\n"
        f"Cited passages:\n\n{passages or '(none)'}"
    )


def free_scores(results: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    merged = results.merge(
        questions[["question_id", "answer", "question_type", "evidence_pages"]].rename(
            columns={"answer": "gold_answer"}
        ),
        on="question_id",
    )
    merged["numeric_match"] = [
        numeric_match(g, a) if t == "metrics-generated" else None
        for g, a, t in zip(merged["gold_answer"], merged["answer"], merged["question_type"], strict=True)
    ]
    merged["cites_evidence"] = [
        cites_evidence(c, e) for c, e in zip(merged["cited_chunk_ids"], merged["evidence_pages"], strict=True)
    ]
    return merged


def judge_run(
    run_path: Path,
    questions: pd.DataFrame,
    contexts_dir: Path,
    judge: ClaudeGenerator,
    max_usd: float,
) -> Path:
    """Add the judge's grades to each record of a run, in ``<run>.judged.jsonl``; resumable."""
    records = [json.loads(line) for line in run_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    contexts = pd.read_parquet(contexts_dir / f"{records[0]['contexts']}.parquet")
    out_path = run_path.with_suffix(".judged.jsonl")
    done = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["question_id"]] = r
    spent = sum(r.get("judge_cost_usd", 0.0) for r in done.values())
    q_by_id = questions.set_index("question_id")
    with out_path.open("a", encoding="utf-8") as out:
        for record in records:
            if record["question_id"] in done:
                continue
            if spent >= max_usd:
                print(f"\nstopped: judge spending reached {spent:.2f} USD")
                break
            q = q_by_id.loc[record["question_id"]]
            cited = contexts[
                (contexts["question_id"] == record["question_id"])
                & contexts["chunk_id"].isin(record["cited_chunk_ids"])
            ].sort_values("position")
            text, response, _ = judge.complete(
                JUDGE_PROMPT, judge_message(q["question"], q["answer"], record["answer"], cited), JUDGE_SCHEMA
            )
            grade = json.loads(text) if response.stop_reason != "refusal" else {}
            price = cost_usd(judge.model, response.usage.input_tokens, response.usage.output_tokens)
            spent += price
            record.update(
                {
                    "judge_model": judge.model,
                    "judge_correctness": grade.get("correctness", "error"),
                    "judge_supported": grade.get("supported"),
                    "judge_reason": grade.get("reason", ""),
                    "judge_cost_usd": round(price, 6),
                }
            )
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[record["question_id"]] = record
            print(f"\r{len(done)}/{len(records)} graded, {spent:.2f} USD", end="", flush=True)
    print()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("runs", nargs="*", type=Path, help="run files; default: every run without grades")
    parser.add_argument("--model", default="claude-opus-5-5")
    parser.add_argument("--effort", default="low")
    parser.add_argument("--max-usd", type=float, required=True, help="spending limit per run")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--contexts", type=Path, default=CONTEXTS_DIR)
    args = parser.parse_args()

    runs = args.runs or sorted(p for p in RUNS_DIR.glob("*.jsonl") if not p.name.endswith(".judged.jsonl"))
    questions = pd.read_parquet(args.questions)
    judge = ClaudeGenerator(args.model, args.effort, max_tokens=4000)
    for run_path in runs:
        print(run_path.name)
        judge_run(run_path, questions, args.contexts, judge, args.max_usd)


if __name__ == "__main__":
    main()
