"""Record the dataset."""
import cv2
import mediapipe as mp
import csv
import os

def init_dataset(filepath: str) -> None:
    if not os.path.exists(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        columns = []
        for i in range(21):
            columns.extend([f'x{i}', f'y{i}', f'z{i}'])
        columns.append('label')
        with open(filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        print(f'Created new dataset file: {filepath}')
    else:
        print(f'Dataset file already exists: {filepath}')

def extract_landmarks(hand_landmarks) -> list:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def save_sample(filepath, coords, label):
    row = coords + [label]
    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def collect_sign(label: str, target_count: int=100, filepath: str='data/raw/psl_data.csv'):
    init_dataset(filepath)
    saved_count = 0
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    while cap.isOpened() and saved_count < target_count:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        key = cv2.waitKey(1) & 255
        if key == ord('q'):
            break
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = extract_landmarks(hand_landmarks)
                if key == ord(' '):
                    save_sample(filepath, coords, label)
                    saved_count = saved_count + 1
        cv2.putText(frame, f'Sign: {label} | Saved: {saved_count}/{target_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Hold hand & press SPACE to record | 'q' to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow('Data Collector', frame)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Finished! Recorded {saved_count} samples for '{label}'.")
    return saved_count
SIGNS = ['alif', 'bay', 'tay', 'say', 'chay', 'daal', 'ray', 'ain', 'laam', 'yae', 'pay', 'hey', 'khey', 'ddal', 'zhe', 'seen', 'zuaad', 'zoe', 'fey', 'qaaf', 'kaaf', 'gaaf', 'meem', 'noon', 'wao', 'choti_yae', 'ttay', 'zaal', 'zay', 'sheen', 'suaad', 'toyen', 'ghain', 'noon_ghunna', 'do_chashmi_hey', 'hamza']
URDU_MAP = {'alif': 'ا', 'bay': 'ب', 'tay': 'ت', 'say': 'ث', 'chay': 'چ', 'daal': 'د', 'ray': 'ر', 'ain': 'ع', 'laam': 'ل', 'yae': 'ے', 'pay': 'پ', 'hey': 'ح', 'khey': 'خ', 'ddal': 'ڈ', 'zhe': 'ژ', 'seen': 'س', 'zuaad': 'ض', 'zoe': 'ظ', 'fey': 'ف', 'qaaf': 'ق', 'kaaf': 'ک', 'gaaf': 'گ', 'meem': 'م', 'noon': 'ن', 'wao': 'و', 'choti_yae': 'ی', 'ttay': 'ٹ', 'zaal': 'ذ', 'zay': 'ز', 'sheen': 'ش', 'suaad': 'ص', 'toyen': 'ط', 'ghain': 'غ', 'noon_ghunna': 'ں', 'do_chashmi_hey': 'ھ', 'hamza': 'ء'}

def _get_recorded_counts(filepath):
    counts = {}
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        lbl = row[-1]
                        counts[lbl] = counts.get(lbl, 0) + 1
        except Exception:
            pass
    return counts

def main():
    while True:
        counts = _get_recorded_counts('data/raw/psl_data.csv')
        total_samples = sum(counts.values())
        print('\n=======================================================')
        print(f'       PSL DATASET COLLECTOR MENU (Total: {total_samples} samples)       ')
        print('=======================================================')
        for idx, sign in enumerate(SIGNS, 1):
            char = URDU_MAP.get(sign, '')
            n = counts.get(sign, 0)
            status = f'({n:2d} samples)' if n > 0 else ' [NEW]'
            print(f'{idx:2d}. {sign:<15} ({char})  {status}')
        print('-------------------------------------------------------')
        print(' A. Record ALL signs in sequence')
        print(' Q. Quit')
        choice = input(f'\nSelect an option (1-{len(SIGNS)}, A, or Q): ').strip().upper()
        if choice == 'Q':
            print('Exiting data collector. Happy learning!')
            break
        if choice == 'A':
            count_input = input('How many samples per sign? (Press Enter for default 20): ').strip()
            count = int(count_input) if count_input.isdigit() else 20
            for sign in SIGNS:
                input(f"\n--> Prepare hand for '{sign}' ({URDU_MAP.get(sign, '')}). Press ENTER to start recording...")
                collect_sign(sign, target_count=count)
            print('\nSuccessfully finished recording sequence for all signs!')
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(SIGNS):
            selected_sign = SIGNS[int(choice) - 1]
            count_input = input(f"How many samples for '{selected_sign}' ({URDU_MAP.get(selected_sign, '')})? (Press Enter for default 20): ").strip()
            count = int(count_input) if count_input.isdigit() else 20
            collect_sign(selected_sign, target_count=count)
        else:
            print(f'Invalid selection. Please choose a number from 1 to {len(SIGNS)}, A, or Q.')
if __name__ == '__main__':
    main()
