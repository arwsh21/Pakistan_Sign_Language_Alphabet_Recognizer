"""Automated tests for HandSample, Environments, SignAgent, and WordBuilder."""

import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure imports from src work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents import SignAgent, WordBuilder
from src.envs import RecordedEnvironment
from src.features import HandSample


class SequenceMockModel:
    """Mock model that returns sequential predictions from a preset list."""

    def __init__(self, sequence: list, proba_val: float = 0.95):
        self.sequence = list(sequence)
        self.cursor = 0
        self.proba_val = proba_val

    def predict(self, X):
        if self.cursor < len(self.sequence):
            pred = self.sequence[self.cursor]
            self.cursor += 1
            return [pred]
        return [self.sequence[-1]]

    def proba(self, X):
        return np.array([[self.proba_val]])


# ==============================================================================
# TEST 1: Normalisation Invariance to Shift (Translation)
# ==============================================================================
def test_normalise_shift_invariance():
    """Shifting all coordinates by a fixed offset must produce the identical normalized vector."""
    # Base raw coordinates (21 points)
    base_raw = []
    for i in range(21):
        base_raw.extend([0.1 * i, 0.05 * i, 0.02 * i])

    # Shifted coordinates (+0.3 to all x, y, z)
    shifted_raw = [val + 0.3 for val in base_raw]

    sample_base = HandSample(base_raw)
    sample_shifted = HandSample(shifted_raw)

    vec_base = sample_base.tovector()
    vec_shifted = sample_shifted.tovector()

    assert np.allclose(vec_base, vec_shifted, atol=1e-6), (
        "Normalised feature vectors must be invariant to translation/shift!"
    )


# ==============================================================================
# TEST 2: Normalisation Invariance to Scale (Size)
# ==============================================================================
def test_normalise_scale_invariance():
    """Multiplying all coordinates by 2 must produce the identical normalized vector."""
    base_raw = []
    for i in range(21):
        base_raw.extend([0.05 * i, 0.1 * i, 0.01 * i])

    # Scaled coordinates (* 2.0 to all points)
    scaled_raw = [val * 2.0 for val in base_raw]

    sample_base = HandSample(base_raw)
    sample_scaled = HandSample(scaled_raw)

    vec_base = sample_base.tovector()
    vec_scaled = sample_scaled.tovector()

    assert np.allclose(vec_base, vec_scaled, atol=1e-6), (
        "Normalised feature vectors must be invariant to scale/hand size!"
    )


# ==============================================================================
# TEST 3: SignAgent Majority Vote Smoothing
# ==============================================================================
def test_sign_agent_smoothing_majority_vote():
    """A rogue prediction in the middle (e.g. bay among alifs) must be filtered out by majority vote."""
    # Sequence with a rogue guess: alif, alif, bay, alif, alif
    mock_seq = SequenceMockModel(sequence=["alif", "alif", "bay", "alif", "alif"])
    agent = SignAgent(model=mock_seq, memory_size=5, confidence_threshold=0.6)

    dummy_features = [0.1] * 60
    final_sign = None
    for _ in range(5):
        final_sign, _ = agent.act(dummy_features)

    # alif has 4 votes vs bay's 1 vote -> alif must win!
    assert final_sign == "alif", f"Expected 'alif' to win majority vote, got '{final_sign}'"


# ==============================================================================
# TEST 4: SignAgent Low Confidence Returns 'unsure'
# ==============================================================================
def test_sign_agent_low_confidence_unsure():
    """Predictions with confidence below 0.6 threshold must return 'unsure'."""
    low_conf_mock = SequenceMockModel(sequence=["alif"], proba_val=0.3)
    agent = SignAgent(model=low_conf_mock, memory_size=5, confidence_threshold=0.6)

    dummy_features = [0.1] * 60
    smoothed, conf = agent.act(dummy_features)

    assert smoothed == "unsure", f"Expected 'unsure' when confidence is 0.3, got '{smoothed}'"
    assert conf == 0.3


# ==============================================================================
# TEST 5: WordBuilder No Repeat
# ==============================================================================
def test_word_builder_no_repeat():
    """Holding for 16 frames adds the letter on frame 15, and does NOT add it again on frame 16."""
    wb = WordBuilder(hold_frames=15)

    # First 14 frames: no letter added
    for _ in range(14):
        added = wb.update("bay")
        assert added is False

    # 15th frame: locks in letter
    added = wb.update("bay")
    assert added is True
    assert wb.get_word() == "bay"

    # 16th frame: holding continues, must NOT duplicate
    added = wb.update("bay")
    assert added is False
    assert wb.get_word() == "bay", "Letter must not be duplicated on frame 16!"


# ==============================================================================
# TEST 6: WordBuilder Streak Reset on 'no sign'
# ==============================================================================
def test_word_builder_streak_reset():
    """Holding for 10 frames, sending 'no sign', then holding for 5 frames must NOT add a letter."""
    wb = WordBuilder(hold_frames=15)

    # Hold 10 frames
    for _ in range(10):
        wb.update("alif")

    # Send 'no sign' (hand dropped or lost)
    wb.update("no sign")

    # Hold 5 more frames
    for _ in range(5):
        wb.update("alif")

    # Total frames = 10 + 1 + 5 = 16, but consecutive streak was only 5 -> no letter added!
    assert wb.get_word() == "", "Streak must reset when 'no sign' is received!"
    assert wb.get_urdu_word() == ""


# ==============================================================================
# TEST 7: RecordedEnvironment Sequential Replay
# ==============================================================================
def test_recorded_environment():
    """Verify that RecordedEnvironment replays samples sequentially and ends with None."""
    items = ["sample_1", "sample_2"]
    env = RecordedEnvironment(items)

    assert env.read() == "sample_1"
    assert env.read() == "sample_2"
    assert env.read() is None
