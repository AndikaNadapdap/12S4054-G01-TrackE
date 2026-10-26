"""Sentence-embedding content-based filtering (semantic CBF, Track E).

CONTROLLED COMPARISON with src/tfidf_cbf.py — only the item representation differs:
  Item text   : `content_text_basic` (same field as TF-IDF)
  Item vector : pretrained sentence embedding (Reimers & Gurevych, 2019), L2-normalised
  User profile: mean of the vectors of the user's POSITIVE training items (rating >= 4),
                then L2-normalised (same rule as TF-IDF)
  Score       : cosine similarity (same as TF-IDF)
Item embeddings are cached in `cache/` (git-ignored) so later runs skip encoding.
"""
import hashlib, os
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
from .common import RELEVANCE_THRESHOLD
from .tfidf_cbf import TEXT_FIELD


def sentence_transformer_encoder(model_name, batch_size=64, max_seq_length=None, device=None):
    """Return a function texts -> (n, d) float32 array of L2-normalised embeddings."""
    from sentence_transformers import SentenceTransformer       # imported lazily
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length

    def encode(texts):
        return model.encode(texts, batch_size=batch_size, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    encode.max_seq_length = model.max_seq_length
    return encode


class EmbeddingCBFRecommender:
    def __init__(self, encoder, cache_name=None, cache_dir="cache"):
        self.encoder, self.cache_name, self.cache_dir = encoder, cache_name, cache_dir

    def _item_vectors(self, text):
        key = hashlib.md5("\n".join(text).encode("utf-8")).hexdigest()[:10]
        path = os.path.join(self.cache_dir, f"{self.cache_name}_{key}.npy") if self.cache_name else None
        if path and os.path.exists(path):
            print(f"Loaded cached item embeddings: {path}")
            return np.load(path)
        V = self.encoder(text)
        if path:
            os.makedirs(self.cache_dir, exist_ok=True)
            np.save(path, V)
            print(f"Saved item embeddings to {path}")
        return V

    def fit(self, train, candidates, items):
        self.items = list(candidates)
        self.item_index = {it: j for j, it in enumerate(self.items)}
        text = (items.set_index("item_id")[TEXT_FIELD].reindex(self.items)
                .fillna("").astype(str).tolist())
        # Item metadata only (no interactions) -> no temporal leakage.
        self.V = normalize(self._item_vectors(text))                   # items x dim (L2 rows)

        pos = train[(train.rating >= RELEVANCE_THRESHOLD) & train.item_id.isin(self.item_index)]
        self.users = sorted(pos.user_id.unique())
        self.user_index = {u: k for k, u in enumerate(self.users)}
        rows = pos.user_id.map(self.user_index).values
        cols = pos.item_id.map(self.item_index).values
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(self.users), len(self.items)))
        A.data[:] = 1.0
        A = normalize(A, norm="l1")                                     # row = mean over positives
        self.P = normalize(A @ self.V)                                  # users x dim (L2 rows)
        return self

    def recommend_many(self, users, seen, k=10, batch=2048):
        recs = {}
        for start in range(0, len(users), batch):
            chunk = users[start:start + batch]
            idx = [self.user_index[u] for u in chunk]
            S = self.P[idx] @ self.V.T                                  # cosine similarities
            for row, u in enumerate(chunk):
                for it in seen.get(u, ()):
                    j = self.item_index.get(it)
                    if j is not None:
                        S[row, j] = -np.inf
            top = np.argpartition(-S, k, axis=1)[:, :k]
            for row, u in enumerate(chunk):
                order = top[row][np.argsort(-S[row, top[row]], kind="stable")]
                recs[u] = [self.items[j] for j in order]
        return recs
