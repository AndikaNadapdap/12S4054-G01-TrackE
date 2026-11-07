"""Matrix Factorization baseline: implicit-feedback ALS (Hu, Koren & Volinsky, 2008).

Training signal: positive training interactions only (rating >= 4), treated as
binary preference p_ui = 1 with confidence c_ui = 1 + alpha. All other pairs have
p_ui = 0 and confidence 1.

Written in NumPy/SciPy with deterministic initialization for a fixed seed.
"""

import numpy as np
import scipy.sparse as sp

from .common import RELEVANCE_THRESHOLD


class ALSRecommender:
    def __init__(
        self,
        factors=64,
        reg=0.1,
        alpha=10.0,
        iterations=15,
        seed=42,
        verbose=True,
    ):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self.verbose = verbose

    def fit(self, train, candidates):
        """Train implicit-feedback ALS using positive training interactions."""

        self.items = list(candidates)
        self.item_index = {
            item: index
            for index, item in enumerate(self.items)
        }

        positive = train[
            (train.rating >= RELEVANCE_THRESHOLD)
            & train.item_id.isin(self.item_index)
        ]

        self.users = sorted(train.user_id.unique())
        self.user_index = {
            user: index
            for index, user in enumerate(self.users)
        }

        rows = positive.user_id.map(self.user_index).values
        cols = positive.item_id.map(self.item_index).values

        interaction_matrix = sp.csr_matrix(
            (
                np.ones(len(rows)),
                (rows, cols),
            ),
            shape=(len(self.users), len(self.items)),
        )

        interaction_matrix.data[:] = 1.0
        interaction_matrix_t = interaction_matrix.T.tocsr()

        rng = np.random.default_rng(self.seed)

        user_factors = rng.normal(
            0,
            0.01,
            (interaction_matrix.shape[0], self.factors),
        )

        item_factors = rng.normal(
            0,
            0.01,
            (interaction_matrix.shape[1], self.factors),
        )

        for iteration in range(self.iterations):
            user_factors = self._solve(
                interaction_matrix,
                item_factors,
            )

            item_factors = self._solve(
                interaction_matrix_t,
                user_factors,
            )

            if self.verbose:
                print(
                    f"ALS iteration {iteration + 1}/"
                    f"{self.iterations} completed"
                )

        self.X = user_factors
        self.Y = item_factors

        return self

    def _solve(self, matrix, fixed_factors):
        """Update factor vectors while keeping the other side fixed."""

        k = fixed_factors.shape[1]

        factor_product = fixed_factors.T @ fixed_factors
        regularization = self.reg * np.eye(k)

        output = np.zeros((matrix.shape[0], k))

        for row in range(matrix.shape[0]):
            indices = matrix.indices[
                matrix.indptr[row]:matrix.indptr[row + 1]
            ]

            if len(indices) == 0:
                continue

            row_factors = fixed_factors[indices]

            A = (
                factor_product
                + self.alpha * (row_factors.T @ row_factors)
                + regularization
            )

            b = (1.0 + self.alpha) * row_factors.sum(axis=0)

            output[row] = np.linalg.solve(A, b)

        return output

    def recommend_many(self, users, seen, k=10, batch=2048):
        """Generate Top-K recommendations while excluding seen items."""

        if k <= 0:
            raise ValueError("k must be greater than zero.")

        if len(self.items) < k:
            raise ValueError(
                f"Number of candidate items ({len(self.items)}) "
                f"is smaller than k ({k})."
            )

        recommendations = {}

        for start in range(0, len(users), batch):
            chunk = users[start:start + batch]

            user_vectors = np.vstack([
                self.X[self.user_index[user]]
                if user in self.user_index
                else np.zeros(self.factors)
                for user in chunk
            ])

            scores = user_vectors @ self.Y.T

            for row, user in enumerate(chunk):
                for item in seen.get(user, ()):
                    item_index = self.item_index.get(item)

                    if item_index is not None:
                        scores[row, item_index] = -np.inf

            top_indices = np.argpartition(
                -scores,
                kth=k - 1,
                axis=1,
            )[:, :k]

            for row, user in enumerate(chunk):
                ordered_indices = top_indices[row][
                    np.argsort(
                        -scores[row, top_indices[row]],
                        kind="stable",
                    )
                ]

                recommendations[user] = [
                    self.items[index]
                    for index in ordered_indices
                ]

        return recommendations