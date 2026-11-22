"""Command line menu and live recognition app."""
import os
from pathlib import Path
import sys
import cv2
import mediapipe as mp
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agents import SignAgent, WordBuilder
from src.collect import collect_sign, extract_landmarks, init_dataset
from src.config import CONFIDENCE_THRESHOLD, HOLD_FRAMES, MEMORY_SIZE, RAW_DATA_PATH
from src.envs import CameraEnvironment
from src.evaluate import run_experiments
from src.features import HandSample, load_dataset
from src.models import SVMRecognizer

def run_live_recognition():
    print('\n[INFO] Loading dataset and training best model (SVM)...')
    if not os.path.exists(RAW_DATA_PATH):
        print(f"\n[ERROR] Dataset file not found at '{RAW_DATA_PATH}'.")
        print('Please record signs first using Option 1 (Collect Data).\n')
        return
    X, y = load_dataset(RAW_DATA_PATH)
    model = SVMRecognizer()
    model.fit(X, y)
    num_classes = len(np.unique(y))
    print(f'[INFO] Model trained successfully on {len(X)} samples across {num_classes} signs!')
    agent = SignAgent(model=model, memory_size=MEMORY_SIZE, confidence_threshold=CONFIDENCE_THRESHOLD)
    wb = WordBuilder(hold_frames=HOLD_FRAMES)
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    try:
        env = CameraEnvironment(0)
    except RuntimeError as e:
        print(f'[ERROR] {e}')
        return
    print('\n========================================================')
    print(' LIVE PSL RECOGNIZER RUNNING!')
    print(' - Show your hand to the camera to see live predictions.')
    print(' - Hold a sign steady for 15 frames to add it to the word.')
    print(" - Press 'c' on the video window to CLEAR the word.")
    print(" - Press 'q' on the video window to QUIT.")
    print('========================================================\n')
    while True:
        ret, frame = env.read()
        if not ret or frame is None:
            break
        display_frame = frame.copy()
        h, w, _ = display_frame.shape
        rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        features = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                raw_coords = extract_landmarks(hand_landmarks)
                sample = HandSample(raw_coords)
                features = sample.tovector()
        smoothed_sign, confidence = agent.act(features)
        locked = wb.update(smoothed_sign)
        if locked:
            print(f"[NEW LETTER LOCKED] -> Roman: '{wb.get_word()}' | Urdu: '{wb.get_urdu_word()}'")
        cv2.rectangle(display_frame, (0, 0), (w, 110), (20, 20, 20), -1)
        if smoothed_sign in ('no sign', 'unsure'):
            status_text = f'Sign: {smoothed_sign.upper()}'
            sign_color = (128, 128, 128)
        else:
            status_text = f'Sign: {smoothed_sign.upper()} ({confidence * 100:.0f}%)'
            sign_color = (0, 255, 0)
        cv2.putText(display_frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, sign_color, 2)
        bar_x, bar_y, bar_w, bar_h = (20, 55, 200, 15)
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        progress = min(wb.hold_counter / wb.hold_frames, 1.0)
        fill_w = int(progress * bar_w)
        if fill_w > 0:
            bar_color = (0, 255, 100) if progress >= 1.0 else (0, 200, 255)
            cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        if wb.hold_counter == 0:
            hold_text = 'Processing...'
            text_color = (160, 160, 160)
        elif progress >= 1.0:
            hold_text = 'Added!'
            text_color = (0, 255, 100)
        else:
            hold_text = f'Holding: {wb.hold_counter}/{wb.hold_frames}'
            text_color = (220, 220, 220)
        cv2.putText(display_frame, hold_text, (bar_x + bar_w + 12, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)
        cv2.rectangle(display_frame, (0, h - 70), (w, h), (15, 15, 15), -1)
        word_str = wb.get_word() if wb.get_word() else '(empty)'
        cv2.putText(display_frame, f'WORD: {word_str.upper()}', (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, "Keys: 'c' to clear | 'q' to quit", (20, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
        cv2.imshow('Pakistan Sign Language Recognizer', display_frame)
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break
        elif key == ord('c'):
            wb.clear()
            print('[INFO] Word cleared!')
    env.close()
    cv2.destroyAllWindows()
    hands.close()
    print('\nLive session closed.')
    if wb.get_word():
        print(f'Final Spelled Word: {wb.get_word()} | Native Urdu: {wb.get_urdu_word()}')

def main_menu():
    while True:
        print('\n==========================================')
        print('  PAKISTAN SIGN LANGUAGE RECOGNIZER (PSL) ')
        print('==========================================')
        print(' 1. Launch Modern Desktop App (CustomTkinter GUI)')
        print(' 2. Run Lightweight OpenCV Camera Window')
        print(' 3. Run 20-Seed Evaluation & Generate Charts')
        print(' 4. Exit')
        print('==========================================')
        choice = input('Enter your choice (1-4): ').strip()
        if choice == '1':
            try:
                from src.ui import launch_gui
                launch_gui()
            except Exception as e:
                print(f'[ERROR launching GUI] {e}')
                print('Falling back to OpenCV camera mode...')
                run_live_recognition()
        elif choice == '2':
            run_live_recognition()
        elif choice == '3':
            if not os.path.exists(RAW_DATA_PATH):
                print(f"\n[ERROR] Dataset file not found at '{RAW_DATA_PATH}'.")
                print("Please record data first using 'python src/collect.py' before evaluating.\n")
                continue
            X, y = load_dataset(RAW_DATA_PATH)
            df_res, _ = run_experiments(X, y)
            print(df_res.to_string(index=False))
        elif choice == '4':
            print('Exiting. Happy signing!')
            break
        else:
            print('Invalid choice, please enter 1, 2, 3, or 4.')
if __name__ == '__main__':
    main_menu()
