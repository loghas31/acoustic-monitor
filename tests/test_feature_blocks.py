"""Backlog T1.10 / self-review F9 — which feature blocks earn their place.

WHY THIS FILE EXISTS
---------------------
F9 measured that the audio and accel band-ILR blocks are near one-dimensional
on simulated HEALTHY data (participation rank ~1.03 / 1.01 of 8 raw log
fractions) and concluded the 14 band-energy features "carry roughly 2
dimensions of information between them" — worrying, because that is more than
a third of the 37-dim vector. But rank alone does not say whether a block is
USELESS: a block can span almost no directions and still have its one
direction be exactly the one that moves when a fault appears. This file
measures that directly with `tools/feature_block_report.py`'s method (a
per-block Mahalanobis distance, trained on healthy-only windows, scored
against held-out healthy + faulty windows) and pins the two findings that
change what "unproven" should mean for these blocks.

FINDING 1 (refines F9, does not contradict it): despite being near-singular
on healthy data, the band-ILR blocks are NOT along for the ride — their
dominant direction discriminates both a bearing fault and an imbalance fault
at AUC > 0.9. F9's "carries ~2 dimensions of information" is still true and
still means most of the 14 columns are redundant; it does not mean those
columns carry no signal.

FINDING 2 (new, and the more interesting one): the envelope block is FULL
RANK under an imbalance fault (its variance is not suppressed) and yet has
ZERO discriminative power for it (AUC ~ chance). Envelope crest and the
envelope-band ILR coordinates are built to detect impact PERIODICITY — an
imbalance fault has none, it just grows a 1x tone — so the block varies
(matching its severity-linked noise) without that variance being about the
fault at all. This is the T1.5 lesson (effective rank is not an information
measure) confirmed from the opposite direction: a HIGH rank block can also
carry zero information about a specific fault, just as a low-rank block can
carry plenty.

Both findings mean the honest one-line summary for DOC_PIPELINE.md is: no
block on this simulator is "along for the ride" in the sense of contributing
nothing to detection, but which block does the work depends entirely on which
fault you ask about — the thing only a real recording, with real fault modes,
can settle for good.
"""

import numpy as np
import pytest
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score

from capture import SimulatedSource
from features import FEATURE_NAMES, extract_features

FS_A, FS_V = 16000, 6400
WINDOW_S = 8.0

BLOCKS = {
    "audio_stat":     [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("audio_stat_")],
    "accel_stat":     [i for i, n in enumerate(FEATURE_NAMES)
                       if n.startswith("accel_") and "band" not in n],
    "audio_band_ilr": [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("audio_band_ilr_")],
    "accel_band_ilr": [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("accel_band_ilr_")],
    "envelope":       [i for i, n in enumerate(FEATURE_NAMES) if n.startswith("env_")],
}


def _matrix(cases, seed0):
    rows = []
    for i, (kind, sev, fr) in enumerate(cases):
        src = SimulatedSource(WINDOW_S, FS_A, FS_V,
                              lambda j, k=kind, s=sev, f=fr: {"kind": k, "severity": s, "fr": f},
                              seed=seed0 + i)
        audio, accel = next(iter(src.windows()))
        rows.append(extract_features(audio, FS_A, accel, FS_V)["vector"])
    return np.array(rows)


def _healthy_two_speed(n=48):
    return [("normal", 0.0, 50.0 if i % 2 == 0 else 30.0) for i in range(n)]


def _ramp(kind, n=48, sev_max=0.5, fr=50.0):
    return [(kind, sev_max * (i + 1) / n, fr) for i in range(n)]


def _d2(train, test):
    mu, sd = train.mean(0), train.std(0) + 1e-12
    tr, te = (train - mu) / sd, (test - mu) / sd
    lw = LedoitWolf().fit(tr)
    diff = te - lw.location_
    return np.einsum("ij,jk,ik->i", diff, lw.precision_, diff)


# ----------------------------------------------------------------------------
# fixtures — build every matrix ONCE per session, several tests read them
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def train_healthy():
    return _matrix(_healthy_two_speed(48), seed0=30000)


@pytest.fixture(scope="module")
def test_healthy():
    return _matrix(_healthy_two_speed(48), seed0=31000)


@pytest.fixture(scope="module")
def test_bearing():
    return _matrix(_ramp("bearing_outer", 48, 0.5), seed0=32000)


@pytest.fixture(scope="module")
def test_imbalance():
    return _matrix(_ramp("imbalance", 48, 1.0), seed0=33000)


def _auc(train_healthy, test_healthy, test_fault, block_idx):
    y = np.concatenate([np.zeros(len(test_healthy)), np.ones(len(test_fault))])
    d2 = np.concatenate([
        _d2(train_healthy[:, block_idx], test_healthy[:, block_idx]),
        _d2(train_healthy[:, block_idx], test_fault[:, block_idx]),
    ])
    return roc_auc_score(y, d2)


# ----------------------------------------------------------------------------
# the block partition itself
# ----------------------------------------------------------------------------

def test_blocks_partition_the_37_dim_vector():
    """No feature is in two blocks, none is left out. If FEATURE_NAMES grows a
    new prefix this fails loudly instead of silently under-counting a block."""
    all_idx = sorted(sum(BLOCKS.values(), []))
    assert all_idx == list(range(37)), (
        "BLOCKS must exactly partition the 37-dim vector; got "
        f"{len(all_idx)} indices, missing {sorted(set(range(37)) - set(all_idx))}")


# ----------------------------------------------------------------------------
# Finding 1: the band-ILR blocks are near-singular but not uninformative
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("block", ["audio_band_ilr", "accel_band_ilr"])
def test_band_ilr_blocks_are_near_singular_on_healthy_data(block, train_healthy):
    """Corroborates F9 with this file's own data and method (entropy-based
    effective rank on the ILR-transformed 7-column block, not F9's raw 8-column
    log-fraction measurement — the two are different objects, so the numbers
    differ, but the conclusion — near one-dimensional — is the same)."""
    X = train_healthy[:, BLOCKS[block]]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    er = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    assert er < 3.0, (
        f"{block} effective rank is {er:.2f} of 7 — F9's near-singularity "
        "finding may no longer hold; re-check DOC_SELF_REVIEW F9")


@pytest.mark.parametrize("block", ["audio_band_ilr", "accel_band_ilr"])
@pytest.mark.parametrize("fault_fixture", ["test_bearing", "test_imbalance"])
def test_band_ilr_blocks_still_detect_both_fault_kinds(
        block, fault_fixture, train_healthy, test_healthy, request):
    """THE CORE OF FINDING 1. Despite being near one-dimensional on healthy
    data, each band-ILR block's single dominant direction discriminates a
    bearing fault AND an unrelated imbalance fault at AUC > 0.85. F9's "along
    for the ride" reading (implied, not stated) does not survive contact with
    a held-out ROC curve — low rank is not the same as no signal."""
    test_fault = request.getfixturevalue(fault_fixture)
    auc = _auc(train_healthy, test_healthy, test_fault, BLOCKS[block])
    assert auc > 0.85, (
        f"{block} AUC against {fault_fixture} is {auc:.3f} — expected >0.85; "
        "if this newly fails, the block really may be along for the ride "
        "for this fault kind, which would be worth recording")


# ----------------------------------------------------------------------------
# Finding 2: the envelope block is the mirror image — full rank, zero signal,
# for the ONE fault kind it is not built to see
# ----------------------------------------------------------------------------

def test_envelope_block_detects_bearing_faults(train_healthy, test_healthy, test_bearing):
    """Sanity check: the envelope block is where most of the bearing-fault
    detection work happens (the system overview (not in this public copy), DOC_PIPELINE.md). If this regresses,
    something upstream of this file broke the headline claim, not this file."""
    auc = _auc(train_healthy, test_healthy, test_bearing, BLOCKS["envelope"])
    assert auc > 0.95, f"envelope block AUC on bearing fault is only {auc:.3f}"


def test_envelope_block_does_not_detect_imbalance(train_healthy, test_healthy, test_imbalance):
    """THE CORE OF FINDING 2. Envelope crest and the envelope-band ILR
    coordinates measure impact PERIODICITY. An imbalance fault has none — it
    is a growing 1x tone, not a train of impacts — so this block should be
    close to chance (0.5) even though (next test) its variance is not
    suppressed. A block being full rank tells you it varies; it does not tell
    you what it varies WITH."""
    auc = _auc(train_healthy, test_healthy, test_imbalance, BLOCKS["envelope"])
    assert auc < 0.65, (
        f"envelope block AUC on imbalance is {auc:.3f}, expected near-chance "
        "(<0.65) — if this now discriminates imbalance, the envelope crest "
        "feature may be picking up something worth investigating, e.g. the "
        "protrugram band-selector reacting to the growing 1x tone")


def test_envelope_block_is_not_rank_suppressed_under_imbalance(train_healthy, test_imbalance):
    """Completes finding 2: the envelope block's lack of imbalance signal is
    not because imbalance flattens its variance (which would just be a
    different, boring explanation). Effective rank stays close to its
    healthy-data value, so the block is genuinely varying — just not with the
    fault label. Rank and information are different axes, confirmed again
    from the opposite direction to T1.5's."""
    X = test_imbalance[:, BLOCKS["envelope"]]
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-30)
    s = np.linalg.svd(Xc, compute_uv=False)
    p = s / s.sum()
    er = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    assert er > 4.0, (
        f"envelope block effective rank collapsed to {er:.2f} of 7 under "
        "imbalance — that WOULD explain the chance-level AUC directly, which "
        "would be a simpler story than the one this test file tells")


# ----------------------------------------------------------------------------
# Finding 3 (incidental, worth pinning): the two per-channel statistics blocks
# are the most reliably informative blocks tested, for either fault kind
# ----------------------------------------------------------------------------

@pytest.mark.parametrize("block", ["audio_stat", "accel_stat"])
@pytest.mark.parametrize("fault_fixture", ["test_bearing", "test_imbalance"])
def test_channel_statistics_blocks_detect_both_fault_kinds(
        block, fault_fixture, train_healthy, test_healthy, request):
    """RMS/kurtosis/crest/skew is where the least assumption-laden signal
    lives: it needs no band selection, no periodicity, nothing about geometry.
    Both statistics blocks should be strong for both fault kinds tested."""
    test_fault = request.getfixturevalue(fault_fixture)
    auc = _auc(train_healthy, test_healthy, test_fault, BLOCKS[block])
    assert auc > 0.9, f"{block} AUC against {fault_fixture} is only {auc:.3f}"
