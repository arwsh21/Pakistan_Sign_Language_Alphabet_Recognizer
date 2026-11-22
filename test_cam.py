import cv2
import mediapipe as mp
from src.collect import extract_landmarks

def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not access webcam (index 0). Make sure your camera is connected and not in use by another app.')
        return
    print('==================================================')
    print('Webcam connected! Show your hand to the camera.')
    print('Printing 21 landmark points in real-time.')
    print("Press 'q' on the camera window to exit.")
    print('==================================================\n')
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Failed to capture video frame.')
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_coords = extract_landmarks(hand_landmarks)
                print(f'Extracted {len(raw_coords)} numbers successfully!')
                print(f'First 3 (Wrist): {raw_coords[:3]}')
        cv2.imshow("PSL Recognizer - Hand Tracking (Press 'q' to quit)", frame)
        if cv2.waitKey(1) & 255 == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print('\nCamera released. Test finished.')
if __name__ == '__main__':
    main()
