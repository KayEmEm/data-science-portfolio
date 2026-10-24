"""Movie recommendations demo: pick a test user, see the history and the top 10 of each model.

Run from the project folder after ``make train``: ``make demo``, or
``streamlit run demo/app.py -- --data-dir data --artifacts artifacts``.
"""

import argparse
import sys
from pathlib import Path

import streamlit as st

from movie_recs.config import ARTIFACTS_DIR, DATA_DIR
from movie_recs.data import load
from movie_recs.demo import load_lists_by_model, user_view
from movie_recs.interactions import make_stage
from movie_recs.report import LABELS


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS_DIR)
    return parser.parse_args(sys.argv[1:])


@st.cache_resource
def resources(data_dir: Path, artifacts: Path):
    data = load(data_dir)
    stage = make_stage(data["ratings"], data["split"], "test")
    return stage, data["movies"], load_lists_by_model(artifacts)


def main() -> None:
    args = arguments()
    st.set_page_config(page_title="Movie recommendations", layout="wide")
    st.title("Movie recommendations: test window")
    stage, movies, lists = resources(args.data_dir, args.artifacts)
    users = lists["ranker"][0]
    user = st.selectbox("User", users, index=0)
    view = user_view(stage, movies, lists, int(user))
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Last rated before the window")
        st.dataframe(view["history"], hide_index=True)
        st.subheader("Liked in the window")
        st.write(", ".join(view["liked_in_window"]) or "none")
    with right:
        st.subheader("Top 10 per model (* = liked in the window)")
        st.dataframe(view["recommendations"].rename(columns=LABELS))


main()
