"""Gradio demo: ask a question about one filing or all filings, read the answer and its cited passages.

Needs the indexes from ``make index``. Answers come from the Claude API when
``ANTHROPIC_API_KEY`` is set (each question is a paid call), from a local
Ollama model when ``--ollama-model`` is given, and otherwise from the
extractive baseline.
"""

import argparse

import pandas as pd
from portfolio_common.config import parse_args

from filings_qa.config import CHUNKS_PATH, DOCUMENTS_PATH, INDEX_DIR, RetrievalSettings
from filings_qa.generation import ExtractiveGenerator, make_generator
from filings_qa.retrieval import load_retrievers

ALL_FILINGS = "All filings"


def build_app(generator, retrievers: dict, chunks: pd.DataFrame, documents: pd.DataFrame):
    import gradio as gr

    by_id = chunks.set_index("chunk_id")
    settings = RetrievalSettings()

    def answer(question: str, filing: str, retriever_name: str):
        if not question.strip():
            return "Type a question.", ""
        doc_name = None if filing == ALL_FILINGS else filing
        hits = retrievers[retriever_name].search(question, settings.top_k, doc_name)
        passages = by_id.loc[[chunk_id for chunk_id, _ in hits]].reset_index()
        passages["position"] = range(1, len(passages) + 1)
        g = generator.generate(question, passages)
        cited = "\n\n".join(
            f"**[{i}] {passages.iloc[i - 1]['company']} {passages.iloc[i - 1]['doc_type']} "
            f"{passages.iloc[i - 1]['doc_period']}, page {passages.iloc[i - 1]['page'] + 1}**\n\n"
            f"{passages.iloc[i - 1]['text']}"
            for i in g.citations
        )
        footer = (
            f"\n\n*{g.input_tokens} input and {g.output_tokens} output tokens, "
            f"{g.latency_s:.1f} s, {g.cost_usd:.4f} USD*"
        )
        return g.answer + footer, cited or "No passage cited."

    with gr.Blocks(title="Filings question answering") as app:
        gr.Markdown(f"## Question answering over company filings\nAnswers by `{generator.model}`.")
        with gr.Row():
            question = gr.Textbox(label="Question", lines=2, scale=3)
            filing = gr.Dropdown(
                [ALL_FILINGS, *sorted(documents["doc_name"])], value=ALL_FILINGS, label="Filing"
            )
            retriever = gr.Radio(list(retrievers), value="dense", label="Retriever")
        ask = gr.Button("Answer", variant="primary")
        result = gr.Markdown()
        sources = gr.Markdown()
        ask.click(answer, [question, filing, retriever], [result, sources])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--claude-model", default=None, help="Claude model id; needs ANTHROPIC_API_KEY")
    parser.add_argument("--ollama-model", default=None, help="Ollama model tag, for example qwen2.5:7b")
    parser.add_argument("--port", type=int, default=7860)
    args = parse_args(parser, __file__)

    if args.claude_model:
        generator = make_generator("claude", args.claude_model, "low")
    elif args.ollama_model:
        generator = make_generator("ollama", args.ollama_model, "low")
    else:
        generator = ExtractiveGenerator()
    app = build_app(
        generator, load_retrievers(INDEX_DIR), pd.read_parquet(CHUNKS_PATH), pd.read_parquet(DOCUMENTS_PATH)
    )
    app.launch(server_port=args.port)


if __name__ == "__main__":
    main()
