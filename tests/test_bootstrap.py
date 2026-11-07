import numpy as np
import pytest

from src.bootstrap import paired_bootstrap


def test_positive_difference():
    a = np.array([0.8, 0.9, 0.7, 0.9])
    b = np.array([0.5, 0.6, 0.4, 0.5])

    result = paired_bootstrap(a, b)

    assert result["mean_difference"] > 0
    assert result["significant"] is True


def test_negative_difference():
    a = np.array([0.2, 0.3, 0.2, 0.1])
    b = np.array([0.7, 0.8, 0.6, 0.7])

    result = paired_bootstrap(a, b)

    assert result["mean_difference"] < 0
    assert result["significant"] is True


def test_different_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


def test_empty_scores():
    with pytest.raises(ValueError):
        paired_bootstrap([], [])


def test_reproducibility():
    a = [0.5, 0.6, 0.7, 0.8]
    b = [0.3, 0.4, 0.5, 0.6]

    result1 = paired_bootstrap(a, b, seed=42)
    result2 = paired_bootstrap(a, b, seed=42)

    assert result1 == result2