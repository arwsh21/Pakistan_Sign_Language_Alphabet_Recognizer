"""Automated tests for HandSample, Environments, SignAgent, and WordBuilder."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agents import SignAgent, WordBuilder
from src.envs import RecordedEnvironment
from src.features import HandSample

class SequenceMockModel:

    def __init__(self, sequence: list, proba_val: float=0.95):
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

def test_normalise_shift_invariance():
    base_raw = []
    for i in range(21):
        base_raw.extend([0.1 * i, 0.05 * i, 0.02 * i])
    shifted_raw = [val + 0.3 for val in base_raw]
    sample_base = HandSample(base_raw)
    sample_shifted = HandSample(shifted_raw)
    vec_base = sample_base.tovector()
    vec_shifted = sample_shifted.tovector()
    assert np.allclose(vec_base, vec_shifted, atol=1e-06), 'Normalised feature vectors must be invariant to translation/shift!'

def test_normalise_scale_invariance():
    base_raw = []
    for i in range(21):
        base_raw.extend([0.05 * i, 0.1 * i, 0.01 * i])
    scaled_raw = [val * 2.0 for val in base_raw]
    sample_base = HandSample(base_raw)
    sample_scaled = HandSample(scaled_raw)
    vec_base = sample_base.tovector()
    vec_scaled = sample_scaled.tovector()
    assert np.allclose(vec_base, vec_scaled, atol=1e-06), 'Normalised feature vectors must be invariant to scale/hand size!'

def test_sign_agent_smoothing_majority_vote():
    mock_seq = SequenceMockModel(sequence=['alif', 'alif', 'bay', 'alif', 'alif'])
    agent = SignAgent(model=mock_seq, memory_size=5, confidence_threshold=0.6)
    dummy_features = [0.1] * 60
    final_sign = None
    for _ in range(5):
        final_sign, _ = agent.act(dummy_features)
    assert final_sign == 'alif', f"Expected 'alif' to win majority vote, got '{final_sign}'"

def test_sign_agent_low_confidence_unsure():
    low_conf_mock = SequenceMockModel(sequence=['alif'], proba_val=0.3)
    agent = SignAgent(model=low_conf_mock, memory_size=5, confidence_threshold=0.6)
    dummy_features = [0.1] * 60
    smoothed, conf = agent.act(dummy_features)
    assert smoothed == 'unsure', f"Expected 'unsure' when confidence is 0.3, got '{smoothed}'"
    assert conf == 0.3

def test_word_builder_no_repeat():
    wb = WordBuilder(hold_frames=15)
    for _ in range(14):
        added = wb.update('bay')
        assert added is False
    added = wb.update('bay')
    assert added is True
    assert wb.get_word() == 'bay'
    added = wb.update('bay')
    assert added is False
    assert wb.get_word() == 'bay', 'Letter must not be duplicated on frame 16!'

def test_word_builder_streak_reset():
    wb = WordBuilder(hold_frames=15)
    for _ in range(10):
        wb.update('alif')
    wb.update('no sign')
    for _ in range(5):
        wb.update('alif')
    assert wb.get_word() == '', "Streak must reset when 'no sign' is received!"
    assert wb.get_urdu_word() == ''

def test_recorded_environment():
    items = ['sample_1', 'sample_2']
    env = RecordedEnvironment(items)
    assert env.read() == 'sample_1'
    assert env.read() == 'sample_2'
    assert env.read() is None
