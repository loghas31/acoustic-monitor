"""
Inference plumbing tests (v2). Scoring logic itself is covered by
test_baseline.py; this file covers the failure modes and the v1.5 guard.
"""

from pathlib import Path

import numpy as np
import pytest

from baseline import fit_baseline, save_baseline
from features import FEATURE_NAMES
from inference import MahalanobisScorer

# The scorer now validates that a baseline's stored dimension matches the
# firmware's feature contract (T1.5 moved it from 40 to 37), so synthetic
# baselines in tests must be built at the current width rather than a literal.
D = len(FEATURE_NAMES)


def test_missing_baseline_is_loud():
    """A device without a learn period must refuse to score — silently scoring
    against nothing would mean silently alerting on nothing."""
    with pytest.raises(FileNotFoundError):
        MahalanobisScorer(Path("/nonexistent/baseline.npz"))


def test_scorer_output_contract(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 40                                   # windows, independent of D
    X = rng.normal(0, 1, (n, D))
    OP = np.column_stack([50 + rng.normal(0, 0.1, n),
                          rng.normal(-0.1, 0.01, n), rng.normal(-0.1, 0.01, n)])
    save_baseline(tmp_path / "b.npz", fit_baseline(X, OP))
    s = MahalanobisScorer(tmp_path / "b.npz").score(rng.normal(0, 1, D),
                                                    np.array([50.0, -0.1, -0.1]))
    assert set(s) == {"score", "regime", "threshold", "anomalous"}
    assert s["score"] >= 0 and s["threshold"] > 0
    assert isinstance(s["anomalous"], bool)


def test_cloud_ae_scorer_optional():
    """v1.5 path: must be importable without TensorFlow installed, and only
    fail (cleanly) when actually constructed."""
    from inference import CloudAEScorer  # import itself must never require TF
    try:
        import tflite_runtime  # noqa: F401
        has_tfl = True
    except ImportError:
        try:
            import tensorflow  # noqa: F401
            has_tfl = True
        except ImportError:
            has_tfl = False
    if not has_tfl:
        with pytest.raises(Exception):
            CloudAEScorer(Path("/nonexistent/model.tflite"), {"lo": 0, "hi": 1})
