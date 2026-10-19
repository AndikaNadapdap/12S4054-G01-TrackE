"""Matrix Factorization baseline: implicit-feedback ALS (Hu, Koren & Volinsky, 2008).

Training signal: positive training interactions only (rating >= 4), treated as
binary preference p_ui = 1 with confidence c_ui = 1 + alpha. All other pairs have
p_ui = 0 and confidence 1. Alternating closed-form updates:

    x_u = (Y^T Y + alpha * Y_u^T Y_u + reg * I)^-1  (1 + alpha) * Y_u^T 1
    y_i = (X^T X + alpha * X_i^T X_i + reg * I)^-1  (1 + alpha) * X_i^T 1

Written in plain NumPy/SciPy so that results are deterministic for a fixed seed.
"""
import numpy as np
import scipy.sparse as sp
from .common import RELEVANCE_THRESHOLD


class ALSRecommender:
    def __init__(self, factors=64, reg=0.1, alpha=10.0, iterations=15, seed=42, verbose=True):
        self.factors, self.reg, self.alpha = factors, reg, alpha
        self.iterations, self.seed, self.verbose = iterations, seed, verbose

    # ---------------------------------------------------------------- fit
    def fit(self, train, candidates):
        self.items = list(candidates)
        self.item_index = {it: j for j, it in enumerate(self.items)}
        pos = train[(train.rating >= RELEVANCE_THRESHOLD) & train.item_id.isin(self.item_index)]
        self.users = sorted(train.user_id.unique())
        self.user_index = {u: k for k, u in enumerate(self.users)}

        rows = pos.user_id.map(self.user_index).values
        cols = pos.item_id.map(self.item_index).values
        R = sp.csr_matrix((np.ones(len(rows)), (rows, cols)),
                          shape=(len(self.users), len(self.items)))
        R.data[:] = 1.0                                    # binary preference
        Rt = R.T.tocsr()

        rng = np.random.default_rng(self.seed)
        X = rng.normal(0, 0.01, (R.shape[0], self.factors))
        Y = rng.normal(0, 0.01, (R.shape[1], self.factors))
        for it in range(self.iterations):
            X = self._solve(R, Y)
            Y = self._solve(Rt, X)
            if self.verbose:
                print(f"  ALS iteration {it + 1}/{self.iterations} done")
        self.X, self.Y = X, Y
        return self

    def _solve(self, M, F):
        """Update every row of M's side given the fixed factor matrix F."""
        k = F.shape[1]
        FtF = F.T @ F
        reg_I = self.reg * np.eye(k)
        out = np.zeros((M.shape[0], k))
        for r in range(M.shape[0]):
            idx = M.indices[M.indptr[r]:M.indptr[r + 1]]
            if len(idx) == 0:
                continue                                    # no positives -> zero vector
            Fr = F[idx]
            A = FtF + self.alpha * (Fr.T @ Fr) + reg_I
            b = (1.0 + self.alpha) * Fr.sum(axis=0)
            out[r] = np.linalg.solve(A, b)
        return out

    # ---------------------------------------------------------- recommend
    def recommend_many(self, users, seen, k=10, batch=2048):
        """Top-k item_ids for each user, excluding items seen in training."""
        recs = {}
        for start in range(0, len(users), batch):
            chunk = users[start:start + batch]
            U = np.vstack([self.X[self.user_index[u]] for u in chunk])
            S = U @ self.Y.T
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
