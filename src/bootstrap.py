"""Paired bootstrap significance testing for Track E CP3."""

import numpy as np


def paired_bootstrap(
    scores_a,
    scores_b,
    n_resamples=1000,
    confidence_level=0.95,
    seed=42,
):
    """
    Estimate confidence interval of paired mean differences.

    Positive difference means model A performs better.
    Negative difference means model B performs better.
    """

    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)

    if scores_a.shape != scores_b.shape:
        raise ValueError(
            "Both score arrays must have the same shape."
        )

    if scores_a.ndim != 1:
        raise ValueError(
            "Score arrays must be one-dimensional."
        )

    if len(scores_a) == 0:
        raise ValueError(
            "Score arrays cannot be empty."
        )

    if (
        not np.isfinite(scores_a).all()
        or not np.isfinite(scores_b).all()
    ):
        raise ValueError(
            "Score arrays must contain only finite values."
        )

    if n_resamples < 1:
        raise ValueError(
            "n_resamples must be positive."
        )

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    differences = scores_a - scores_b
    observed_difference = differences.mean()

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_resamples)

    for i in range(n_resamples):
        sample_indices = rng.integers(
            0,
            len(differences),
            size=len(differences),
        )

        bootstrap_means[i] = (
            differences[sample_indices].mean()
        )

    alpha = 1 - confidence_level

    ci_lower = np.quantile(
        bootstrap_means,
        alpha / 2,
    )

    ci_upper = np.quantile(
        bootstrap_means,
        1 - alpha / 2,
    )

    return {
        "mean_difference": float(observed_difference),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_resamples": n_resamples,
        "n_users": len(differences),
        "seed": seed,
        "significant": bool(
            ci_lower > 0 or ci_upper < 0
        ),
    }