import json
from types import SimpleNamespace

import pandas as pd
import pytest

from filings_qa import generation
from filings_qa.generation import (
    ClaudeGenerator,
    ExtractiveGenerator,
    Generation,
    OllamaGenerator,
    cost_usd,
    parse_answer,
    run,
)
from filings_qa.scoring import cites_evidence, numeric_match, parse_numbers


@pytest.mark.parametrize(
    ("gold", "answer", "expected"),
    [
        ("$1577.00", "Capital expenditure was $1,577 million in FY2018.", True),
        ("$1577.00", "It was $1.58 billion.", True),
        ("$1577.00", "It was $1,600 million.", False),
        ("0.66", "The quick ratio is 0.66.", True),
        ("0.66", "Margin was 66%.", True),
        ("-3.7", "Growth was (3.7)%.", True),
        ("-3.7", "Growth was 3.7%.", False),
        ("24.26", "24.3", True),
        ("Yes, it did.", "Yes", None),
    ],
)
def test_numeric_match(gold, answer, expected):
    assert numeric_match(gold, answer) is expected


def test_parse_numbers_reads_separators_and_parentheses():
    assert parse_numbers("$1,234.50 and (12)") == [(1234.5, 2), (-12.0, 0)]


def test_cites_evidence_compares_pages_not_passages():
    assert cites_evidence(["A#3#2"], ["A#3"])
    assert not cites_evidence(["A#4#0"], ["A#3"])
    assert not cites_evidence([], ["A#3"])


def test_parse_answer_drops_citations_outside_the_passages():
    answer, citations, answerable = parse_answer(
        json.dumps({"answer": "x", "citations": [0, 2, 2, 9], "answerable": True}), 3
    )
    assert (answer, citations, answerable) == ("x", [2], True)
    assert parse_answer("not json", 3) == ("not json", [], True)


def test_cost_uses_the_price_table():
    assert cost_usd("claude-haiku-5-5", 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert cost_usd("local-model", 10, 10) == 0.0


def _passages(n=3):
    return pd.DataFrame(
        {
            "question_id": ["q"] * n,
            "position": range(1, n + 1),
            "chunk_id": [f"A#{i}#0" for i in range(n)],
            "company": "A",
            "doc_name": "A",
            "doc_type": "10K",
            "doc_period": 2022,
            "page": range(n),
            "text": [f"passage {i}" for i in range(n)],
        }
    )


def test_claude_generator_reads_answer_usage_and_cost():
    message = SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking"),
            SimpleNamespace(type="text", text='{"answer": "7", "citations": [2], "answerable": true}'),
        ],
        usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
        stop_reason="end_turn",
    )
    calls = []
    client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: calls.append(kw) or message))
    g = object.__new__(ClaudeGenerator)
    g.client, g.model, g.effort, g.max_tokens = client, "claude-opus-5-5", "low", 100
    result = g.generate("What?", _passages())
    assert (result.answer, result.citations, result.answerable) == ("7", [2], True)
    assert result.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)
    assert calls[0]["output_config"]["effort"] == "low"
    assert "[2] A, 10K 2022, page 2" in calls[0]["messages"][0]["content"]


def test_claude_generator_requires_a_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        ClaudeGenerator()


def test_ollama_generator_posts_the_schema_and_reads_counts(monkeypatch):
    sent = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "message": {"content": '{"answer": "a", "citations": [1], "answerable": true}'},
                    "prompt_eval_count": 50,
                    "eval_count": 5,
                    "done_reason": "stop",
                }
            ).encode()

    def fake_urlopen(request, timeout):
        sent["url"] = request.full_url
        sent["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(generation.urllib.request, "urlopen", fake_urlopen)
    result = OllamaGenerator("qwen2.5:7b", url="localhost:11434").generate("What?", _passages())
    assert sent["url"] == "http://localhost:11434/api/chat"
    assert sent["body"]["format"]["required"] == ["answer", "citations", "answerable"]
    assert (result.input_tokens, result.output_tokens, result.cost_usd) == (50, 5, 0.0)


class PaidStub:
    name, model = "stub", "stub"

    def __init__(self):
        self.calls = 0

    def generate(self, question, passages):
        self.calls += 1
        return Generation("a", [1], True, 10, 1, 0.1, 0.4)


def test_run_stops_at_the_spending_limit_and_resumes(tmp_path):
    questions = pd.DataFrame({"question_id": ["q", "r", "s"], "question": ["x", "y", "z"]})
    contexts = pd.concat([_passages().assign(question_id=q) for q in "qrs"])
    out = tmp_path / "run.jsonl"
    stub = PaidStub()
    first = run(stub, questions, contexts, out, {"run": "r"}, max_usd=0.5)
    assert len(first) == 2 and stub.calls == 2
    second = run(stub, questions, contexts, out, {"run": "r"}, max_usd=10)
    assert len(second) == 3 and stub.calls == 3
    assert second.iloc[0]["cited_chunk_ids"] == ["A#0#0"]


def test_extractive_answers_with_the_first_passage():
    g = ExtractiveGenerator().generate("q", _passages())
    assert (g.answer, g.citations) == ("passage 0", [1])
