"""TF-IDF content-based filtering (lexical CBF baseline, Track E).

Item text   : `content_text_basic` from items.csv (the SAME field will be used by the
              sentence-embedding model, so only the representation differs).
Item vector : TF-IDF, L2-normalised (Salton & Buckley, 1988).
User profile: mean of the vectors of the user's POSITIVE training items (rating >= 4),
              then L2-normalised. Built from training data only.
Score       : cosine similarity between user profile and every candidate item.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from .common import RELEVANCE_THRESHOLD

TEXT_FIELD = "content_text_basic"


class TfidfCBFRecommender:
    def __init__(self, ngram_range=(1, 1), min_df=2, max_df=0.5, max_features=None,
                 sublinear_tf=True, stop_words="english"):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df,
                                          max_features=max_features, sublinear_tf=sublinear_tf,
                                          stop_words=stop_words, lowercase=True,
                                          dtype=np.float32)

    def fit(self, train, candidates, items):
        self.items = list(candidates)
        self.item_index = {it: j for j, it in enumerate(self.items)}
        text = (items.set_index("item_id")[TEXT_FIELD].reindex(self.items)
                .fillna("").astype(str).tolist())
        # Item metadata only (no interactions) -> no temporal leakage.
        self.V = normalize(self.vectorizer.fit_transform(text))          # items x vocab (L2 rows)

        pos = train[(train.rating >= RELEVANCE_THRESHOLD) & train.item_id.isin(self.item_index)]
        self.users = sorted(pos.user_id.unique())
        self.user_index = {u: k for k, u in enumerate(self.users)}
        rows = pos.user_id.map(self.user_index).values
        cols = pos.item_id.map(self.item_index).values
        A = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                          shape=(len(self.users), len(self.items)))
        A.data[:] = 1.0
        A = normalize(A, norm="l1")                                     # row = mean over positives
        self.P = normalize(A @ self.V)                                  # users x vocab (L2 rows)
        return self

    def recommend_many(self, users, seen, k=10, batch=2048):
        recs = {}
        VT = self.V.T.tocsr()
        for start in range(0, len(users), batch):
            chunk = users[start:start + batch]
            idx = [self.user_index[u] for u in chunk]
            S = (self.P[idx] @ VT).toarray()                            # cosine similarities
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

    @property
    def vocab_size(self):
        return len(self.vectorizer.vocabulary_)
