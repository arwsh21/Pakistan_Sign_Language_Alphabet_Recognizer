"""Camera and recorded input."""
import cv2

class Environment:

    def read(self):
        raise NotImplementedError()

    def close():
        pass

class CameraEnvironment(Environment):

    def __init__(self, camera_index: int=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f'Could not open webcam at index {camera_index}. Check camera connection.')

    def read(self):
        if not self.cap.isOpened():
            return (False, None)
        return self.cap.read()

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

class RecordedEnvironment(Environment):

    def __init__(self, samples: list):
        self.samples = list(samples)
        self._cursor = 0

    def read(self):
        if self._cursor < len(self.samples):
            sample = self.samples[self._cursor]
            self._cursor += 1
            return sample
        return None

    def close(self):
        self._cursor = 0
if __name__ == '__main__':
    dummy_data = ['sample_A', 'sample_B', 'sample_C']
    rec_env = RecordedEnvironment(dummy_data)
    print('Testing RecordedEnvironment replay:')
    print('1st read:', rec_env.read())
    print('2nd read:', rec_env.read())
    print('3rd read:', rec_env.read())
    print('4th read:', rec_env.read())
    try:
        cam_env = CameraEnvironment(0)
        ret, frame = cam_env.read()
        print(f'\nCamera opened successfully! Frame captured: {ret}')
        cam_env.close()
    except RuntimeError as e:
        print(f'\nCamera check: {e}')
