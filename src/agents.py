"""Real time agent and word builder."""

from collections import Counter, deque
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIDENCE_THRESHOLD, HOLD_FRAMES, MEMORY_SIZE

URDU_CHAR_MAP = {
    # Initial 10 signs
    "alif": "ا",
    "bay": "ب",
    "tay": "ت",
    "say": "ث",
    "chay": "چ",
    "daal": "د",
    "ray": "ر",
    "ain": "ع",
    "laam": "ل",
    "yae": "ے",
    # Extended alphabet signs
    "pay": "پ",
    "hey": "ح",
    "khey": "خ",
    "ddal": "ڈ",
    "zhe": "ژ",
    "seen": "س",
    "zuaad": "ض",
    "zoe": "ظ",
    "fey": "ف",
    "qaaf": "ق",
    "kaaf": "ک",
    "gaaf": "گ",
    "meem": "م",
    "noon": "ن",
    "wao": "و",
    "choti_yae": "ی",
    # Final batch (excluding motion signs)
    "ttay": "ٹ",
    "zaal": "ذ",
    "zay": "ز",
    "sheen": "ش",
    "suaad": "ص",
    "toyen": "ط",
    "ghain": "غ",
    "noon_ghunna": "ں",
    "do_chashmi_hey": "ھ",
    "hamza": "ء",
}


class SignAgent:
  """Model-based reflex agent with internal memory buffer for smoothing predictions.

  Maintains a private sliding window of the last 10 predictions and takes a
  majority vote.
  """

  def __init__(
      self,
      model,
      memory_size: int = MEMORY_SIZE,
      confidence_threshold: float = CONFIDENCE_THRESHOLD,
  ):
    self.model = model
    self.confidence_threshold = confidence_threshold

    self._memory = deque(maxlen=memory_size)

  def act(self, features):
    """Takes 60 normalised coordinates (or None if no hand detected).

    Returns a tuple of (smoothed_sign, confidence).
    """
    if features is None:
      # No hand visible in the frame
      self._memory.append("no sign")
      return "no sign", 0.0

    feat_arr = np.array(features).reshape(1, -1)

    pred_label = self.model.predict(feat_arr)[0]
    confidence = 1.0

    if hasattr(self.model, "proba"):
      probs = self.model.proba(feat_arr)
      if probs is not None:
        confidence = float(np.max(probs))
    elif hasattr(self.model.model, "predict_proba"):
      probs = self.model.model.predict_proba(feat_arr)
      confidence = float(np.max(probs))

    if confidence < self.confidence_threshold:
      raw_prediction = "unsure"
    else:
      raw_prediction = pred_label

    self._memory.append(raw_prediction)

    # Majority vote over the last 10 frames
    smoothed_sign = Counter(self._memory).most_common(1)[0][0]

    return smoothed_sign, confidence

  def reset(self):
    """Clears the memory buffer."""
    self._memory.clear()


class WordBuilder:
  """Builds words by locking in a letter after it has been held stably for 15 frames."""

  def __init__(self, hold_frames: int = HOLD_FRAMES):
    self.hold_frames = hold_frames
    self.current_candidate = None
    self.hold_counter = 0
    self.letters = []  
    self.urdu_letters = []  
    
  def update(self, smoothed_sign: str) -> bool:
    """Updates the letter counter with the latest smoothed sign from the agent.

    Returns True if a new letter was locked into the word, else False.
    """
    # If no hand or unsure, reset hold progress
    if smoothed_sign in ("no sign", "unsure"):
      self.current_candidate = None
      self.hold_counter = 0
      return False

    # If holding the same sign as before
    if smoothed_sign == self.current_candidate:
      self.hold_counter += 1
      # Lock in the letter when we hit exactly hold_frames (e.g. 15 frames)
      if self.hold_counter == self.hold_frames:
        self.letters.append(smoothed_sign)
        urdu_char = URDU_CHAR_MAP.get(smoothed_sign, "")
        if urdu_char:
          self.urdu_letters.append(urdu_char)
        return True
    else:
      # Hand switched to a new sign shape
      self.current_candidate = smoothed_sign
      self.hold_counter = 1

    return False

  def get_word(self) -> str:
    """Returns the word spelled in Roman Urdu (e.g. 'bay laam')."""
    return " ".join(self.letters)

  def get_urdu_word(self) -> str:
    """Returns the word in native conjoined Urdu script (e.g. 'بل')."""
    return "".join(self.urdu_letters)

  def clear(self):
    """Resets the word back to empty."""
    self.letters.clear()
    self.urdu_letters.clear()
    self.current_candidate = None
    self.hold_counter = 0


if __name__ == "__main__":
  # Quick test of WordBuilder logic!
  wb = WordBuilder(hold_frames=15)

  print("Simulating holding 'bay' for 15 frames...")
  for i in range(14):
    wb.update("bay")
  print(f"Frame 14 word: '{wb.get_word()}' (Still waiting for frame 15)")

  locked = wb.update("bay")  # Frame 15!
  print(f"Frame 15 locked in? {locked}")
  print(f"Roman Word: '{wb.get_word()}'")
  print(f"Native Urdu Word: '{wb.get_urdu_word()}'")

  print("\nSimulating holding 'laam' for 15 frames...")
  for i in range(15):
    wb.update("laam")
  print(f"Roman Word: '{wb.get_word()}'")
  print(f"Native Urdu Word: '{wb.get_urdu_word()}' (Notice: it conjoined to بل!)")