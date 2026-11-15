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
            columns.extend([f"x{i}", f"y{i}", f"z{i}"])
        columns.append("label")

        with open(filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
        print(f"Created new dataset file: {filepath}")
    else:
        print(f"Dataset file already exists: {filepath}")


def extract_landmarks(hand_landmarks) -> list:
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def save_sample(filepath, coords, label):
    row = coords + [label]
    with open(filepath, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)

def collect_sign(label: str, target_count: int = 100, filepath: str = "data/raw/psl_data.csv"):
    # 1. PREPARATION
    init_dataset(filepath)  
    saved_count = 0          

    # 2. START CAMERa and MP
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

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                coords = extract_landmarks(hand_landmarks)
                if key == ord(' '):
                    save_sample(filepath, coords, label)
                    saved_count = saved_count +1        

        # 5. SHOW STATUS ON SCREEN
        cv2.putText(frame, f"Sign: {label} | Saved: {saved_count}/{target_count}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Hold hand & press SPACE to record | 'q' to quit", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Data Collector", frame)

    # 6. CLEANUP (Exact same as test_cam.py)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print(f"Finished! Recorded {saved_count} samples for '{label}'.")
    return saved_count


SIGNS = [
    "alif",
    "bay",
    "tay",
    "say",
    "chay",
    "daal",
    "ray",
    "ain",
    "laam",
    "yae",
]


def main():
    """Interactive command-line menu for collecting PSL alphabet dataset."""
    while True:
        print("\n===============================")
        print("   PSL DATASET COLLECTOR MENU   ")
        print("===============================")
        for idx, sign in enumerate(SIGNS, 1):
            print(f"{idx:2d}. {sign}")
        print(" A. Record ALL 10 signs in sequence")
        print(" Q. Quit")

        choice = input("\nSelect an option (1-10, A, or Q): ").strip().upper()

        if choice == "Q":
            print("Exiting data collector. Happy learning!")
            break

        if choice == "A":
            count_input = input("How many samples per sign? (Press Enter for default 20): ").strip()
            count = int(count_input) if count_input.isdigit() else 20
            for sign in SIGNS:
                input(f"\n--> Prepare hand for '{sign}'. Press ENTER to start recording...")
                collect_sign(sign, target_count=count)
            print("\nSuccessfully finished recording sequence for all signs!")
            continue

        if choice.isdigit() and 1 <= int(choice) <= len(SIGNS):
            selected_sign = SIGNS[int(choice) - 1]
            count_input = input(f"How many samples for '{selected_sign}'? (Press Enter for default 20): ").strip()
            count = int(count_input) if count_input.isdigit() else 20
            collect_sign(selected_sign, target_count=count)
        else:
            print("Invalid selection. Please choose a number from 1 to 10, A, or Q.")


if __name__ == "__main__":
    main()
