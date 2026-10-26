"""Answer each question from its retrieved passages, with citations, and record tokens, cost and latency.

Generators:

* ``extractive``: no language model; returns the first retrieved passage. A floor.
* ``claude``: the Claude API (paid; needs ``ANTHROPIC_API_KEY``).
* ``ollama``: an open weights model served by a local Ollama (no API cost).

Output: one JSON line per question in ``artifacts/runs/<run>.jsonl``. A run
resumes where it stopped, and stops before its spending reaches ``--max-usd``.
"""

import argparse
import json
import os
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from portfolio_common.config import parse_args

from filings_qa.config import (
    ANTHROPIC_API_URL,
    API_KEY_ENV,
    OLLAMA_DEFAULT_URL,
    OLLAMA_URL_ENV,
    PRICES_USD_PER_MTOK,
    QUESTIONS_PATH,
    RUNS_DIR,
)

SYSTEM_PROMPT = """You answer questions from financial analysts about company filings.
Use only the numbered passages in the user message; they are extracts of the filings.
If the passages do not contain what the answer needs, say so and set "answerable" to false.
Show the calculation when the answer is computed from several figures.
Return JSON with these fields:
- "answer": the answer in one to three sentences, with figures, units and the period they refer to.
- "citations": the numbers of the passages that support the answer.
- "answerable": true when the passages contain what the answer needs."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "integer"}},
        "answerable": {"type": "boolean"},
    },
    "required": ["answer", "citations", "answerable"],
    "additionalProperties": False,
}


@dataclass
class Generation:
    answer: str
    citations: list[int]
    answerable: bool
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    stop_reason: str = ""
    raw: str = field(default="", repr=False)


def format_passages(passages: pd.DataFrame) -> str:
    """Number the passages from 1 and label each with company, form, period and page (one based)."""
    blocks = []
    for i, p in enumerate(passages.itertuples(index=False), start=1):
        blocks.append(f"[{i}] {p.company}, {p.doc_type} {p.doc_period}, page {p.page + 1}\n{p.text}")
    return "\n\n".join(blocks)


def user_message(question: str, passages: pd.DataFrame) -> str:
    return f"Passages:\n\n{format_passages(passages)}\n\nQuestion: {question}"


def parse_answer(text: str, n_passages: int) -> tuple[str, list[int], bool]:
    """Read the JSON answer; keep only citation numbers that name a passage."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return text.strip(), [], True
    citations = sorted({int(c) for c in data.get("citations", []) if 1 <= int(c) <= n_passages})
    return str(data.get("answer", "")).strip(), citations, bool(data.get("answerable", True))


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if model not in PRICES_USD_PER_MTOK:
        return 0.0
    price_in, price_out = PRICES_USD_PER_MTOK[model]
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


class ExtractiveGenerator:
    """Returns the first passage unchanged and cites it. Free and fast; says nothing it did not read."""

    name = "extractive"
    model = "none"

    def generate(self, question: str, passages: pd.DataFrame) -> Generation:
        if passages.empty:
            return Generation("No passage was retrieved.", [], False)
        return Generation(passages.iloc[0]["text"], [1], True)


class ClaudeGenerator:
    name = "claude"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", max_tokens: int = 8000):
        import anthropic

        key = os.environ.get(API_KEY_ENV)
        if not key:
            raise RuntimeError(f"set {API_KEY_ENV} to use the Claude API")
        self.client = anthropic.Anthropic(api_key=key, base_url=ANTHROPIC_API_URL, max_retries=4)
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens

    def complete(self, system: str, content: str, schema: dict) -> tuple[str, object, float]:
        start = time.perf_counter()
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        latency = time.perf_counter() - start
        text = next((b.text for b in response.content if b.type == "text"), "")
        return text, response, latency

    def generate(self, question: str, passages: pd.DataFrame) -> Generation:
        text, response, latency = self.complete(
            SYSTEM_PROMPT, user_message(question, passages), ANSWER_SCHEMA
        )
        usage = response.usage
        if response.stop_reason == "refusal":
            answer, citations, answerable = "", [], False
        else:
            answer, citations, answerable = parse_answer(text, len(passages))
        return Generation(
            answer,
            citations,
            answerable,
            usage.input_tokens,
            usage.output_tokens,
            latency,
            cost_usd(self.model, usage.input_tokens, usage.output_tokens),
            str(response.stop_reason),
            text,
        )


class OllamaGenerator:
    """Calls the chat endpoint of a local Ollama server with the answer schema as output format."""

    name = "ollama"

    def __init__(self, model: str, url: str | None = None, context_tokens: int = 16384):
        self.model = model
        self.url = (url or os.environ.get(OLLAMA_URL_ENV) or OLLAMA_DEFAULT_URL).rstrip("/")
        if not self.url.startswith("http"):
            self.url = f"http://{self.url}"
        self.context_tokens = context_tokens

    def generate(self, question: str, passages: pd.DataFrame) -> Generation:
        body = {
            "model": self.model,
            "stream": False,
            "format": ANSWER_SCHEMA,
            "options": {"temperature": 0, "seed": 42, "num_ctx": self.context_tokens},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(question, passages)},
            ],
        }
        request = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with urllib.request.urlopen(request, timeout=900) as response:
            data = json.loads(response.read())
        latency = time.perf_counter() - start
        text = data["message"]["content"]
        answer, citations, answerable = parse_answer(text, len(passages))
        return Generation(
            answer,
            citations,
            answerable,
            int(data.get("prompt_eval_count", 0)),
            int(data.get("eval_count", 0)),
            latency,
            0.0,
            str(data.get("done_reason", "")),
            text,
        )


def make_generator(kind: str, model: str | None, effort: str):
    if kind == "extractive":
        return ExtractiveGenerator()
    if kind == "claude":
        return ClaudeGenerator(model or "claude-opus-5-5", effort)
    if kind == "ollama":
        if not model:
            raise ValueError("--model is required for ollama, for example qwen2.5:7b")
        return OllamaGenerator(model)
    raise ValueError(f"unknown generator {kind}")


def estimate_usd(model: str, contexts: pd.DataFrame, questions: pd.DataFrame, output_tokens: int) -> dict:
    """Rough cost before any call: four characters per token, ``output_tokens`` per answer.

    The output figure includes the model's thinking tokens, which are billed
    as output. It is an assumption; the real figure is in each run's records.
    """
    chars = len(SYSTEM_PROMPT) * len(questions)
    for q in questions.itertuples(index=False):
        chars += len(user_message(q.question, contexts[contexts["question_id"] == q.question_id]))
    input_tokens = chars // 4
    total_output = output_tokens * len(questions)
    return {
        "questions": len(questions),
        "input_tokens": input_tokens,
        "output_tokens": total_output,
        "usd": round(cost_usd(model, input_tokens, total_output), 2),
    }


def read_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    generator,
    questions: pd.DataFrame,
    contexts: pd.DataFrame,
    out_path: Path,
    run_info: dict,
    max_usd: float,
) -> pd.DataFrame:
    done = read_done(out_path)
    finished = {r["question_id"] for r in done}
    spent = sum(r["cost_usd"] for r in done)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    by_question = dict(tuple(contexts.groupby("question_id")))
    with out_path.open("a", encoding="utf-8") as out:
        for q in questions.itertuples(index=False):
            if q.question_id in finished:
                continue
            if spent >= max_usd:
                print(f"\nstopped: spending reached {spent:.2f} USD, the limit is {max_usd:.2f} USD")
                break
            passages = by_question.get(q.question_id, contexts.iloc[0:0]).sort_values("position")
            g = generator.generate(q.question, passages)
            spent += g.cost_usd
            cited = [passages.iloc[i - 1]["chunk_id"] for i in g.citations]
            record = {
                **run_info,
                "question_id": q.question_id,
                "answer": g.answer,
                "citations": g.citations,
                "cited_chunk_ids": cited,
                "answerable": g.answerable,
                "input_tokens": g.input_tokens,
                "output_tokens": g.output_tokens,
                "latency_s": round(g.latency_s, 3),
                "cost_usd": round(g.cost_usd, 6),
                "stop_reason": g.stop_reason,
            }
            out.write(json.dumps(record) + "\n")
            out.flush()
            done.append(record)
            print(f"\r{len(done)}/{len(questions)} answered, {spent:.2f} USD", end="", flush=True)
    print()
    return pd.DataFrame(done)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--contexts", type=Path, required=True, help="a file written by evaluate_retrieval")
    parser.add_argument("--generator", choices=["extractive", "claude", "ollama"], required=True)
    parser.add_argument("--model", help="Claude model id or Ollama model tag")
    parser.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--limit", type=int, help="answer only the first N questions")
    parser.add_argument("--max-usd", type=float, default=0.0, help="spending limit for paid generators")
    parser.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
    parser.add_argument("--assumed-output-tokens", type=int, default=1500)
    args = parse_args(parser, __file__)

    questions = pd.read_parquet(args.questions)
    if args.limit:
        questions = questions.head(args.limit)
    contexts = pd.read_parquet(args.contexts)
    context_name = args.contexts.stem
    if args.estimate:
        model = args.model or "claude-opus-5-5"
        print(
            json.dumps(
                {"model": model, **estimate_usd(model, contexts, questions, args.assumed_output_tokens)}
            )
        )
        return
    generator = make_generator(args.generator, args.model, args.effort)
    if args.generator == "claude" and args.max_usd <= 0:
        raise SystemExit("set --max-usd for paid runs")
    max_usd = args.max_usd if args.generator == "claude" else float("inf")
    model_tag = generator.model.replace(":", "-").replace("/", "-")
    suffix = f"_{args.effort}" if args.generator == "claude" else ""
    run_name = f"{context_name}__{generator.name}_{model_tag}{suffix}"
    info = {
        "run": run_name,
        "contexts": context_name,
        "generator": generator.name,
        "model": generator.model,
        "effort": args.effort if args.generator == "claude" else "",
    }
    results = run(generator, questions, contexts, args.runs_dir / f"{run_name}.jsonl", info, max_usd)
    print(
        f"{run_name}: {len(results)} answers, {results['cost_usd'].sum():.2f} USD, "
        f"median latency {results['latency_s'].median():.2f} s"
    )


if __name__ == "__main__":
    main()
