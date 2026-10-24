"""Two tower retrieval model in PyTorch, with exact nearest neighbour search in FAISS.

* Item tower: a learned id embedding, the genres, the release decade and the
  ten most frequent user tags of the movie, through a small MLP. In training a
  share of the ids is hidden, so that the features alone place a movie in the
  space. A movie without ratings before the cutoff gets the "unknown" id and
  is placed by its features only (the cold start case).
* User tower: the mean embedding of the user's last ``history`` movies and the
  mean genre profile of those movies, through a small MLP.
* Training: each example is one movie the user rated 4 or more, with the
  movies the user rated before it as the history. The loss is a softmax over
  the other movies of the batch, with the log of each movie's sampling
  frequency subtracted (Yi et al. 2019), so that popular movies are not
  pushed down for being frequent negatives.

Both towers end in an L2 normalization; the score is the cosine similarity.
"""

import json
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from movie_recs.config import GENRES, POSITIVE_RATING, TwoTowerSettings
from movie_recs.interactions import Stage
from movie_recs.metrics import PAD

TAGS_PER_ITEM = 10
DECADES = np.arange(1900, 2040, 10)


class ItemFeatures:
    """Feature rows: 0 is padding, 1..n the catalog items of the stage, n+1.. extra (new) items."""

    def __init__(self, stage: Stage, movies: pd.DataFrame, tags: pd.DataFrame, n_tags: int, extra=()):
        self.catalog = stage.items
        self.extra = np.setdiff1d(np.asarray(extra, dtype=self.catalog.dtype), self.catalog)
        self.item_ids = np.concatenate([[PAD], self.catalog, self.extra])
        meta = movies.set_index("item").reindex(self.item_ids)
        self.genres = meta[GENRES].fillna(0).to_numpy(np.float32)
        years = meta["year"].to_numpy()
        self.decade = np.where(np.isnan(years), 0, np.searchsorted(DECADES, years, side="right")).astype(
            np.int64
        )
        self.ids = np.zeros(len(self.item_ids), dtype=np.int64)  # 0: unknown id
        self.ids[1 : len(self.catalog) + 1] = np.arange(1, len(self.catalog) + 1)
        self.tags, self.vocabulary = self._tags(tags.loc[tags["ts"] < stage.start], n_tags)

    def _tags(self, tags: pd.DataFrame, n_tags: int) -> tuple[np.ndarray, list[str]]:
        vocabulary = tags["tag"].value_counts().index[:n_tags].tolist()
        index = {t: i + 1 for i, t in enumerate(vocabulary)}
        out = np.zeros((len(self.item_ids), TAGS_PER_ITEM), dtype=np.int64)
        kept = tags.loc[tags["tag"].isin(index)]
        counts = kept.groupby(["item", "tag"]).size().rename("n").reset_index()
        counts = counts.sort_values(["item", "n", "tag"], ascending=[True, False, True])
        counts["slot"] = counts.groupby("item").cumcount()
        counts = counts.loc[counts["slot"] < TAGS_PER_ITEM]
        rows = pd.Index(self.item_ids).get_indexer(counts["item"])
        keep = rows > 0
        out[rows[keep], counts["slot"].to_numpy()[keep]] = counts["tag"].map(index).to_numpy()[keep]
        return out, vocabulary

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "genres": torch.tensor(self.genres),
            "decade": torch.tensor(self.decade),
            "ids": torch.tensor(self.ids),
            "tags": torch.tensor(self.tags),
        }


def masked_mean(embedded: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    mask = (index > 0).unsqueeze(-1).float()
    return (embedded * mask).sum(1) / mask.sum(1).clamp(min=1.0)


def mlp(inputs: int, dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, 2 * dim), nn.ReLU(), nn.Linear(2 * dim, dim))


class TwoTower(nn.Module):
    def __init__(self, features: ItemFeatures, settings: TwoTowerSettings):
        super().__init__()
        dim = settings.dim
        for name, tensor in features.tensors().items():
            self.register_buffer(name, tensor, persistent=False)
        n_ids = len(features.catalog) + 1
        self.id_dropout = settings.id_dropout
        self.item_id = nn.Embedding(n_ids, dim)
        self.item_genres = nn.Linear(len(GENRES), dim)
        self.item_decade = nn.Embedding(len(DECADES) + 1, dim)
        self.item_tags = nn.Embedding(len(features.vocabulary) + 1, dim, padding_idx=0)
        self.item_mlp = mlp(4 * dim, dim)
        self.user_history = nn.Embedding(n_ids, dim, padding_idx=0)
        self.user_genres = nn.Linear(len(GENRES), dim)
        self.user_mlp = mlp(2 * dim, dim)
        for embedding in (self.item_id, self.item_decade, self.item_tags, self.user_history):
            nn.init.normal_(embedding.weight, std=0.05)

    def items(self, rows: torch.Tensor) -> torch.Tensor:
        ids = self.ids[rows]
        if self.training and self.id_dropout > 0:
            ids = torch.where(torch.rand(ids.shape) < self.id_dropout, torch.zeros_like(ids), ids)
        tags = self.tags[rows]
        parts = [
            self.item_id(ids),
            self.item_genres(self.genres[rows]),
            self.item_decade(self.decade[rows]),
            masked_mean(self.item_tags(tags), tags),
        ]
        return F.normalize(self.item_mlp(torch.cat(parts, dim=-1)), dim=-1)

    def users(self, history: torch.Tensor) -> torch.Tensor:
        """``history``: [batch, L] feature rows of catalog items, 0 for padding."""
        genres = masked_mean(self.genres[history], history)
        parts = [masked_mean(self.user_history(history), history), self.user_genres(genres)]
        return F.normalize(self.user_mlp(torch.cat(parts, dim=-1)), dim=-1)


class Sequences:
    """Each history user's rated catalog items in time order, as one flat array with offsets."""

    def __init__(self, stage: Stage):
        history = stage.history.sort_values(["user", "ts", "item"], kind="stable")
        self.users = stage.users
        self.items = np.searchsorted(stage.items, history["item"].to_numpy()) + 1  # feature rows
        self.liked = history["rating"].to_numpy() >= POSITIVE_RATING
        rows = np.searchsorted(stage.users, history["user"].to_numpy())
        self.start = np.searchsorted(rows, np.arange(len(stage.users)))
        self.end = np.searchsorted(rows, np.arange(len(stage.users)), side="right")
        self.row_of = rows

    def histories(self, positions: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
        """The ``length`` items before each position (exclusive), 0 where the user has fewer."""
        index = positions[:, None] - length + np.arange(length)[None, :]
        valid = index >= starts[:, None]
        return np.where(valid, self.items[np.clip(index, 0, None)], 0)

    def last(self, rows: np.ndarray, length: int) -> np.ndarray:
        return self.histories(self.end[rows], self.start[rows], length)

    def sample(self, per_user: int, rng: np.random.Generator) -> np.ndarray:
        """Positions of up to ``per_user`` liked items per user that have at least one item before them."""
        positions = np.flatnonzero(self.liked & (np.arange(len(self.items)) > self.start[self.row_of]))
        key = rng.random(len(positions))
        order = np.lexsort((key, self.row_of[positions]))
        positions = positions[order]
        groups = self.row_of[positions]
        rank = np.arange(len(positions)) - np.searchsorted(groups, groups)
        return rng.permutation(positions[rank < per_user])


def checkpoint_path(directory: Path, epoch: int) -> Path:
    return directory / f"epoch_{epoch:02d}.pt"


def train(
    stage: Stage,
    features: ItemFeatures,
    settings: TwoTowerSettings,
    seed: int,
    epochs: int,
    directory: Path,
    on_epoch=None,
) -> TwoTower:
    """Train for ``epochs`` epochs; every epoch is saved and a stopped run resumes from the last one.

    ``on_epoch(model, epoch) -> dict`` is called after each epoch; its result is stored with the checkpoint.
    """
    torch.manual_seed(seed)
    torch.set_num_threads(settings.threads)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "settings.json").write_text(json.dumps(asdict(settings), indent=2))
    sequences = Sequences(stage)
    model = TwoTower(features, settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.lr, weight_decay=settings.weight_decay)
    done = 0
    for epoch in range(epochs, 0, -1):
        path = checkpoint_path(directory, epoch)
        if path.exists():
            state = torch.load(path, weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            done = epoch
            print(f"two tower: resumed after epoch {epoch} from {path}")
            break
    for epoch in range(done + 1, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        positions = sequences.sample(settings.targets_per_user, rng)
        targets = sequences.items[positions]
        counts = np.bincount(targets, minlength=len(features.item_ids)).astype(np.float64)
        log_q = torch.from_numpy(np.log(np.maximum(counts, 1) / counts.sum()).astype(np.float32))
        starts = sequences.start[sequences.row_of[positions]]
        model.train()
        total, steps = 0.0, 0
        for lo in range(0, len(positions), settings.batch_size):
            batch = slice(lo, lo + settings.batch_size)
            if len(positions[batch]) < 2:
                continue
            history = torch.from_numpy(sequences.histories(positions[batch], starts[batch], settings.history))
            target = torch.from_numpy(targets[batch])
            logits = (
                model.users(history) @ model.items(target).T / settings.temperature - log_q[target][None, :]
            )
            duplicate = (target[:, None] == target[None, :]) & ~torch.eye(len(target), dtype=torch.bool)
            logits = logits.masked_fill(duplicate, float("-inf"))
            loss = F.cross_entropy(logits, torch.arange(len(target)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        record = {"epoch": epoch, "examples": int(len(positions)), "loss": total / max(steps, 1)}
        if on_epoch is not None:
            record.update(on_epoch(model, epoch))
        print(f"two tower epoch {epoch}: {json.dumps(record)}")
        torch.save(
            {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "record": record},
            checkpoint_path(directory, epoch),
        )
    return model


def epoch_records(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("epoch_*.pt"))
    return [torch.load(p, weights_only=False)["record"] for p in paths]


@torch.no_grad()
def embed(model: TwoTower, stage: Stage, users: np.ndarray, settings: TwoTowerSettings, extra: bool = False):
    """User vectors for ``users`` and item vectors for the catalog (or for the extra items)."""
    model.eval()
    sequences = Sequences(stage)
    history = torch.from_numpy(sequences.last(stage.user_rows(users), settings.history))
    user_vectors = torch.cat([model.users(history[i : i + 4096]) for i in range(0, len(history), 4096)])
    n = len(stage.items)
    rows = torch.arange(n + 1, len(model.ids)) if extra else torch.arange(1, n + 1)
    item_vectors = torch.cat([model.items(rows[i : i + 8192]) for i in range(0, len(rows), 8192)])
    return user_vectors.numpy(), item_vectors.numpy()


def recommend(
    model: TwoTower, stage: Stage, users: np.ndarray, n: int, settings: TwoTowerSettings, batch: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Top n catalog items per user by inner product in a flat FAISS index, seen items excluded.

    Each batch of users asks the index for n plus the largest number of seen items in the batch,
    so the result is exact after the seen items are removed.
    """
    user_vectors, item_vectors = embed(model, stage, users, settings)
    faiss.omp_set_num_threads(settings.threads)
    index = faiss.IndexFlatIP(item_vectors.shape[1])
    index.add(np.ascontiguousarray(item_vectors, dtype=np.float32))
    rows = stage.user_rows(users)
    seen_counts = np.diff(stage.matrix.indptr)[rows]
    order = np.argsort(seen_counts, kind="stable")
    n = min(n, len(stage.items))
    out_items = np.full((len(users), n), PAD, dtype=np.int64)
    out_scores = np.full((len(users), n), -np.inf, dtype=np.float32)
    for lo in range(0, len(order), batch):
        chunk = order[lo : lo + batch]
        k = int(min(len(stage.items), n + seen_counts[chunk].max()))
        scores, found = index.search(np.ascontiguousarray(user_vectors[chunk]), k)
        for j, position in enumerate(chunk):
            seen = stage.matrix.indices[
                stage.matrix.indptr[rows[position]] : stage.matrix.indptr[rows[position] + 1]
            ]
            keep = ~np.isin(found[j], seen) & (found[j] >= 0)
            picked = found[j][keep][:n]
            out_items[position, : len(picked)] = stage.items[picked]
            out_scores[position, : len(picked)] = scores[j][keep][:n]
    return out_items, out_scores
