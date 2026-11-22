"""MediaPipe wrapper and normalise."""
import os
import math
import pandas as pd
import numpy as np

class HandSample:

    def __init__(self, raw_coords: list, label: str=None):
        if len(raw_coords) != 63:
            raise ValueError(f'Expected 63 coordinates, got {len(raw_coords)}')
        self.label = label
        self.is_normalised = False
        self._points = []
        for i in range(0, len(raw_coords), 3):
            self._points.append(list(raw_coords[i:i + 3]))

    def normalise(self) -> None:
        wrist_x, wrist_y, wrist_z = self._points[0]
        shifted = []
        for x, y, z in self._points:
            shifted.append([x - wrist_x, y - wrist_y, z - wrist_z])
        max_dist = 0.0
        for x, y, z in shifted:
            dist = math.sqrt(x ** 2 + y ** 2 + z ** 2)
            if dist > max_dist:
                max_dist = dist
        if max_dist == 0.0:
            max_dist = 1.0
        self._points = []
        for x, y, z in shifted:
            self._points.append([x / max_dist, y / max_dist, z / max_dist])
        self.is_normalised = True

    def tovector(self) -> list:
        if not self.is_normalised:
            self.normalise()
        vector = []
        for pt in self._points[1:]:
            vector.extend(pt)
        return vector

def load_dataset(filepath: str):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'Dataset file not found at: {filepath}')
    df = pd.read_csv(filepath)
    X_list = []
    y_list = []
    for row in df.values:
        raw_coords = row[:-1]
        label = row[-1]
        sample = HandSample(raw_coords, label=label)
        X_list.append(sample.tovector())
        y_list.append(label)
    X = np.array(X_list)
    y = np.array(y_list)
    return (X, y)
if __name__ == '__main__':
    dummy_raw = [0.5] * 63
    sample = HandSample(dummy_raw, label='alif')
    print(f'Points before normalising: {len(sample._points)}')
    features = sample.tovector()
    print(f'Features returned by tovector(): {len(features)} numbers')
    print(f'Is normalised: {sample.is_normalised}')
    X, y = load_dataset('data/raw/psl_data.csv')
    print('X shape:', X.shape)
    print('y shape:', y.shape)
