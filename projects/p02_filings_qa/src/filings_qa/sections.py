"""Classify 10-K pages by the section they belong to, from their text alone.

Use: route a question to the right part of a filing, or filter retrieval to
one section. Labels come from the filings' own "Item" headings: a heading
opens a section, and every following page carries that section's label
until the next heading. Table of contents pages (three headings or more) are
skipped, as are pages after Item 15, where exhibits and financial
statements mix. Only pages without any heading enter training and test, so the
heading words never reveal the label.

Models, compared on filings of companies held out from training:

* majority class (baseline);
* TF-IDF word and bigram features with logistic regression;
* MiniLM sentence embeddings (the mean of the page's passage vectors from
  the dense index) with logistic regression. The encoder stays frozen: full
  fine tuning needs model files from Hugging Face, which this build
  environment cannot reach.
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from portfolio_common.plotting import apply_style
from portfolio_common.seed import set_seed
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline

from filings_qa.config import INDEX_DIR, PAGES_PATH, REPORTS_DIR, SEED

HEADING = re.compile(
    r"^\s*item\s+(1a|1b|1c|7a|9a|9b|9c|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16)\b\s*[\.:\-–—|]?",
    re.IGNORECASE | re.MULTILINE,
)
SECTIONS = {
    "1": "business",
    "1A": "risk factors",
    "7": "management discussion",
    "7A": "market risk",
    "8": "financial statements",
}
OTHER = "other"
# Many filers place their financial statements after Item 15 (exhibits and
# schedules) or Item 16, together with the exhibit index, so a page there has
# no reliable label. Those pages are left out.
AMBIGUOUS = {"15", "16"}


def label_pages(pages: pd.DataFrame) -> pd.DataFrame:
    """Return the 10-K pages that carry no heading, each with the section it lies in."""
    rows = []
    for doc_name, doc in pages[pages["doc_type"] == "10K"].groupby("doc_name", sort=False):
        current = None
        for page in doc.sort_values("page").itertuples(index=False):
            items = [m.upper() for m in HEADING.findall(page.text)]
            if len(items) >= 3:
                continue
            if items:
                current = items[-1]
                continue
            if current is None or current in AMBIGUOUS or len(page.text.split()) < 50:
                continue
            rows.append(
                {
                    "doc_name": doc_name,
                    "company": page.company,
                    "page": page.page,
                    "text": page.text,
                    "section": SECTIONS.get(current, OTHER),
                }
            )
    return pd.DataFrame(rows)


def page_embeddings(labeled: pd.DataFrame, index_dir: Path) -> np.ndarray:
    """Mean of the stored MiniLM vectors of each page's passages."""
    import chromadb

    collection = chromadb.PersistentClient(path=str(index_dir / "chroma")).get_collection("filings")
    vectors = {}
    for doc_name in labeled["doc_name"].unique():
        got = collection.get(where={"doc_name": doc_name}, include=["embeddings", "metadatas"])
        for vector, meta in zip(got["embeddings"], got["metadatas"], strict=True):
            vectors.setdefault((doc_name, meta["page"]), []).append(vector)
    return np.vstack(
        [np.mean(vectors[(d, p)], axis=0) for d, p in zip(labeled["doc_name"], labeled["page"], strict=True)]
    )


def evaluate(models: dict, features: dict, labels: np.ndarray, train: np.ndarray, test: np.ndarray):
    rows, predictions = [], {}
    for name, model in models.items():
        x = features[name]
        model.fit(x[train], labels[train])
        predicted = model.predict(x[test])
        predictions[name] = predicted
        rows.append(
            {
                "model": name,
                "macro F1": f1_score(labels[test], predicted, average="macro"),
                "accuracy": accuracy_score(labels[test], predicted),
            }
        )
    return pd.DataFrame(rows), predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pages", type=Path, default=PAGES_PATH)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--skip-embeddings", action="store_true")
    args = parser.parse_args()

    set_seed(SEED)
    labeled = label_pages(pd.read_parquet(args.pages))
    labels = labeled["section"].to_numpy()
    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
    train, test = next(split.split(labeled, labels, groups=labeled["company"]))

    texts = labeled["text"].to_numpy()
    models = {
        "majority class": DummyClassifier(strategy="most_frequent"),
        "TF-IDF + logistic regression": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=100_000, sublinear_tf=True),
            LogisticRegression(max_iter=2000, C=10, class_weight="balanced"),
        ),
    }
    features = {"majority class": texts, "TF-IDF + logistic regression": texts}
    if not args.skip_embeddings:
        models["MiniLM embeddings + logistic regression"] = LogisticRegression(
            max_iter=2000, C=10, class_weight="balanced"
        )
        features["MiniLM embeddings + logistic regression"] = page_embeddings(labeled, args.index_dir)
    results, predictions = evaluate(models, features, labels, train, test)

    counts = labeled.assign(split=np.where(np.isin(np.arange(len(labeled)), test), "test", "train"))
    counts = counts.groupby(["section", "split"]).size().unstack(fill_value=0)
    best = results.sort_values("macro F1").iloc[-1]["model"]
    report = classification_report(labels[test], predictions[best], digits=3)
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "section_classification.md").write_text(
        f"Pages: {len(labeled):,} from {labeled['doc_name'].nunique()} 10-K filings; "
        f"test companies: {labeled.iloc[test]['company'].nunique()} of {labeled['company'].nunique()}.\n\n"
        f"{results.round(3).to_markdown(index=False)}\n\nPages per section:\n\n{counts.to_markdown()}\n\n"
        f"Per class, {best}:\n\n```\n{report}```\n",
        encoding="utf-8",
    )
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        labels[test], predictions[best], normalize="true", values_format=".2f", ax=ax, colorbar=False
    )
    ax.set_title(best)
    ax.grid(False)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    (args.reports / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(args.reports / "figures" / "section_confusion.png")
    plt.close(fig)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
