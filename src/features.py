"""MediaPipe wrapper and normalise."""
import os  
import math
import pandas as pd
import numpy as np

class HandSample:
    """Represents a single hand sample with 21 3D landmarks and normalisation logic."""

    def __init__(self, raw_coords: list, label: str = None):
        if len(raw_coords) != 63:
            raise ValueError(f"Expected 63 coordinates, got {len(raw_coords)}")

        self.label = label
        self.is_normalised = False

        # Group 63 numbers into 21 points of [x, y, z] (private attribute)
        self._points = []
        for i in range(0, len(raw_coords), 3):
            self._points.append(list(raw_coords[i : i + 3]))

    def normalise(self) -> None:
        """
        Normalises the hand shape in two steps:
        1. Translates the wrist (point 0) to (0, 0, 0).
        2. Scales by dividing all points by the maximum distance from the wrist.
        """
        # Step 1: Wrist is point 0
        wrist_x, wrist_y, wrist_z = self._points[0]

        # Shift all 21 points so the wrist becomes (0, 0, 0)
        shifted = []
        for x, y, z in self._points:
            shifted.append([x - wrist_x, y - wrist_y, z - wrist_z])

        # Step 2: Find the maximum distance from the wrist to any point
        max_dist = 0.0
        for x, y, z in shifted:
            # 3D Euclidean distance formula: sqrt(x^2 + y^2 + z^2)
            dist = math.sqrt(x**2 + y**2 + z**2)
            if dist > max_dist:
                max_dist = dist

        # Avoid division by zero if all points are identical
        if max_dist == 0.0:
            max_dist = 1.0

        # Divide each point by max_dist so coordinates are scale-invariant
        self._points = []
        for x, y, z in shifted:
            self._points.append([x / max_dist, y / max_dist, z / max_dist])

        self.is_normalised = True

    def tovector(self) -> list:
        """
        Returns the 60 informative numbers.
        Point 0 (wrist) is now always (0, 0, 0), so we discard it and return points 1 to 20.
        """
        if not self.is_normalised:
            self.normalise()

        # Flatten points 1 through 20 (20 points * 3 coordinates = 60 floats)
        vector = []
        for pt in self._points[1:]:
            vector.extend(pt)
        return vector

def load_dataset(filepath: str):
    """
    Loads raw CSV data, normalises each hand sample using HandSample,
    and returns (X, y) where X has 60 features per sample.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")

    df = pd.read_csv(filepath)

    X_list = []
    y_list = []

    for row in df.values:
        raw_coords = row[:-1]  # first 63 values
        label = row[-1]  # the sign name

        # Normalise the sample using our HandSample class
        sample = HandSample(raw_coords, label=label)
        X_list.append(sample.tovector())  # 60 normalised numbers
        y_list.append(label)

    # Convert to NumPy arrays for scikit-learn
    X = np.array(X_list)
    y = np.array(y_list)

    return X, y

if __name__ == "__main__":
    # Test with dummy data
    dummy_raw = [0.5] * 63  # 63 numbers
    sample = HandSample(dummy_raw, label="alif")

    print(f"Points before normalising: {len(sample._points)}")
    features = sample.tovector()
    print(f"Features returned by tovector(): {len(features)} numbers")
    print(f"Is normalised: {sample.is_normalised}")

    X, y = load_dataset("data/raw/psl_data.csv")
    print("X shape:", X.shape)  # Should be (205, 60)
    print("y shape:", y.shape)  # Should be (205,)