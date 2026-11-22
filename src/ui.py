"""Modern Desktop Graphical User Interface (GUI) for PSL Recognizer using CustomTkinter.

Theme Palette Inspired by Autumn Sky & Crimson Leaves:
- LAPIS LAZULI:     #4C6399 (Deep slate/denim blue)
- JORDY BLUE:       #89BEED (Vibrant sky/periwinkle blue)
- LIGHT CORAL:      #E49592 (Warm coral pink)
- AUBURN:           #A21A1D (Rich crimson red)
- CHOCOLATE COSMOS: #420B1D (Deep wine burgundy / dark slate)
"""
import os
from pathlib import Path
import sys
import tkinter as tk
import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agents import SignAgent, WordBuilder, URDU_CHAR_MAP
from src.config import CONFIDENCE_THRESHOLD, HOLD_FRAMES, MEMORY_SIZE, RAW_DATA_PATH
from src.envs import CameraEnvironment
from src.features import HandSample, load_dataset
from src.models import SVMRecognizer
COLOR_APP_BG = ('#F4F6FA', '#1B1116')
COLOR_SIDEBAR_BG = ('#E8EEF7', '#23151D')
COLOR_CONTAINER_BG = ('#E8EEF7', '#23151D')
COLOR_CARD_BG = ('#FFFFFF', '#2C1A25')
COLOR_CARD_BORDER = ('#CBD7EC', '#4C6399')
COLOR_VIDEO_BG = ('#FFFFFF', '#140D11')
COLOR_LOGO = ('#4C6399', '#89BEED')
COLOR_SUBTITLE = ('#5A687A', '#D2C4D0')
COLOR_SECTION_TITLE = ('#4C6399', '#89BEED')
COLOR_SIGN_MAIN = ('#243356', '#FFFFFF')
COLOR_URDU_CHAR = ('#A21A1D', '#89BEED')
COLOR_CONFIDENCE = ('#4C6399', '#E49592')
COLOR_WORD_URDU = ('#A21A1D', '#89BEED')
COLOR_WORD_ROMAN = ('#243356', '#E49592')
COLOR_STATUS_IDLE = ('#6B7888', '#C3B5C1')
COLOR_STATUS_HOLDING = ('#A21A1D', '#E49592')
COLOR_STATUS_LOCKED = ('#4C6399', '#89BEED')
COLOR_BTN_PRIMARY = ('#4C6399', '#4C6399')
COLOR_BTN_PRIMARY_HOVER = ('#3B4E7A', '#627CB8')
COLOR_BTN_DANGER = ('#A21A1D', '#A21A1D')
COLOR_BTN_DANGER_HOVER = ('#831416', '#C4272B')
COLOR_PROGRESS_FILL = ('#4C6399', '#89BEED')
COLOR_PROGRESS_TRACK = ('#D6E0F0', '#4C6399')
ctk.set_appearance_mode('Dark')
ctk.set_default_color_theme('green')

class PSLApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title('Pakistan Sign Language (PSL) Recognizer')
        self.geometry('1120x740')
        self.minsize(980, 660)
        self.configure(fg_color=COLOR_APP_BG)
        self.is_camera_running = False
        self.env = None
        self.agent = None
        self.wb = None
        self.hands = None
        self.mp_draw = None
        self.mp_hands = None
        self._init_pipeline()
        self._create_layout()
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def _init_pipeline(self):
        if not os.path.exists(RAW_DATA_PATH):
            print(f'[WARNING] Dataset not found at {RAW_DATA_PATH}')
            return
        X, y = load_dataset(RAW_DATA_PATH)
        model = SVMRecognizer()
        model.fit(X, y)
        self.agent = SignAgent(model=model, memory_size=MEMORY_SIZE, confidence_threshold=CONFIDENCE_THRESHOLD)
        self.wb = WordBuilder(hold_frames=HOLD_FRAMES)
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)

    def _create_layout(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=COLOR_SIDEBAR_BG)
        self.sidebar.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.sidebar.grid_rowconfigure(9, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar, text='PSL AI', font=ctk.CTkFont(size=28, weight='bold'), text_color=COLOR_LOGO)
        self.logo_label.grid(row=0, column=0, padx=20, pady=(25, 4))
        self.sub_label = ctk.CTkLabel(self.sidebar, text='Pakistan Sign Language\nAlphabet Recognizer', font=ctk.CTkFont(size=12), text_color=COLOR_SUBTITLE, justify='center')
        self.sub_label.grid(row=1, column=0, padx=20, pady=(0, 25))
        self.btn_camera = ctk.CTkButton(self.sidebar, text='Start Camera', command=self.toggle_camera, height=42, font=ctk.CTkFont(size=14, weight='bold'), fg_color=COLOR_BTN_PRIMARY, hover_color=COLOR_BTN_PRIMARY_HOVER, text_color='#FFFFFF', corner_radius=8)
        self.btn_camera.grid(row=2, column=0, padx=20, pady=10)
        self.btn_clear = ctk.CTkButton(self.sidebar, text='Clear Word', command=self.clear_word, height=38, font=ctk.CTkFont(size=13, weight='bold'), fg_color=COLOR_BTN_DANGER, hover_color=COLOR_BTN_DANGER_HOVER, text_color='#FFFFFF', corner_radius=8)
        self.btn_clear.grid(row=3, column=0, padx=20, pady=10)
        self.theme_label = ctk.CTkLabel(self.sidebar, text='Appearance Mode:', font=ctk.CTkFont(size=12, weight='bold'), text_color=COLOR_SECTION_TITLE)
        self.theme_label.grid(row=5, column=0, padx=20, pady=(25, 5), sticky='w')
        self.theme_menu = ctk.CTkOptionMenu(self.sidebar, values=['Dark', 'Light'], command=self.change_theme, fg_color=COLOR_BTN_PRIMARY, button_color=COLOR_BTN_PRIMARY_HOVER, button_hover_color=COLOR_BTN_PRIMARY_HOVER, text_color='#FFFFFF', corner_radius=8)
        self.theme_menu.grid(row=6, column=0, padx=20, pady=(0, 15))
        self.switch_mirror = ctk.CTkSwitch(self.sidebar, text='Mirror Video View', font=ctk.CTkFont(size=12), text_color=COLOR_SIGN_MAIN, progress_color=COLOR_PROGRESS_FILL)
        self.switch_mirror.grid(row=7, column=0, padx=20, pady=(5, 15), sticky='w')
        self.author_label = ctk.CTkLabel(self.sidebar, text='Areesha Ahmad\n24K-3043 | BSE-5B\nFAST NUCES Karachi', font=ctk.CTkFont(size=11), text_color=COLOR_SUBTITLE, justify='center')
        self.author_label.grid(row=10, column=0, padx=20, pady=20)
        self.main_content = ctk.CTkFrame(self, fg_color='transparent')
        self.main_content.grid(row=0, column=1, sticky='nsew', padx=20, pady=20)
        self.main_content.grid_columnconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_rowconfigure(1, weight=0)
        self.video_frame_card = ctk.CTkFrame(self.main_content, corner_radius=12, fg_color=COLOR_VIDEO_BG, border_color=COLOR_CARD_BORDER, border_width=1)
        self.video_frame_card.grid(row=0, column=0, sticky='nsew', padx=0, pady=0)
        self.video_frame_card.grid_columnconfigure(0, weight=1)
        self.video_frame_card.grid_rowconfigure(0, weight=1)
        self.video_label = ctk.CTkLabel(self.video_frame_card, text="Camera is Off\nClick 'Start Camera' in the sidebar to begin.", font=ctk.CTkFont(size=16, weight='bold'), text_color=COLOR_SUBTITLE)
        self.video_label.grid(row=0, column=0, padx=10, pady=10)
        self.dashboard = ctk.CTkFrame(self.main_content, corner_radius=12, fg_color=COLOR_CONTAINER_BG, border_color=COLOR_CARD_BORDER, border_width=1)
        self.dashboard.grid(row=1, column=0, sticky='ew', padx=0, pady=(15, 0), ipady=6)
        self.dashboard.grid_columnconfigure((0, 1, 2), weight=1)
        self.card_sign = ctk.CTkFrame(self.dashboard, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER, border_width=1, corner_radius=10)
        self.card_sign.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        self.lbl_sign_title = ctk.CTkLabel(self.card_sign, text='DETECTED SIGN', font=ctk.CTkFont(size=11, weight='bold'), text_color=COLOR_SECTION_TITLE)
        self.lbl_sign_title.pack(pady=(8, 0))
        self.lbl_detected_sign = ctk.CTkLabel(self.card_sign, text='—', font=ctk.CTkFont(size=28, weight='bold'), text_color=COLOR_SIGN_MAIN)
        self.lbl_detected_sign.pack(pady=(0, 2))
        self.lbl_confidence = ctk.CTkLabel(self.card_sign, text='Confidence: 0%', font=ctk.CTkFont(size=12, weight='bold'), text_color=COLOR_CONFIDENCE)
        self.lbl_confidence.pack(pady=(0, 8))
        self.card_hold = ctk.CTkFrame(self.dashboard, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER, border_width=1, corner_radius=10)
        self.card_hold.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        self.lbl_hold_title = ctk.CTkLabel(self.card_hold, text='HOLD PROGRESS', font=ctk.CTkFont(size=11, weight='bold'), text_color=COLOR_SECTION_TITLE)
        self.lbl_hold_title.pack(pady=(8, 0))
        self.lbl_hold_status = ctk.CTkLabel(self.card_hold, text='Processing...', font=ctk.CTkFont(size=16, weight='bold'), text_color=COLOR_STATUS_IDLE)
        self.lbl_hold_status.pack(pady=(4, 6))
        self.progress_bar = ctk.CTkProgressBar(self.card_hold, width=180, height=12, progress_color=COLOR_PROGRESS_FILL, fg_color=COLOR_PROGRESS_TRACK)
        self.progress_bar.pack(pady=(0, 10))
        self.progress_bar.set(0.0)
        self.card_word = ctk.CTkFrame(self.dashboard, fg_color=COLOR_CARD_BG, border_color=COLOR_CARD_BORDER, border_width=1, corner_radius=10)
        self.card_word.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')
        self.lbl_word_title = ctk.CTkLabel(self.card_word, text='FORMED WORD', font=ctk.CTkFont(size=11, weight='bold'), text_color=COLOR_SECTION_TITLE)
        self.lbl_word_title.pack(pady=(8, 0))
        self.lbl_urdu_word = ctk.CTkLabel(self.card_word, text='(خالی)', font=ctk.CTkFont(family='Arial', size=30, weight='bold'), text_color=COLOR_WORD_URDU)
        self.lbl_urdu_word.pack(pady=(0, 0))
        self.lbl_roman_word = ctk.CTkLabel(self.card_word, text='Roman: (empty)', font=ctk.CTkFont(size=13, weight='bold'), text_color=COLOR_WORD_ROMAN)
        self.lbl_roman_word.pack(pady=(0, 8))

    def toggle_camera(self):
        if not self.is_camera_running:
            try:
                self._init_pipeline()
                self.env = CameraEnvironment(0)
                self.is_camera_running = True
                self.btn_camera.configure(text='Stop Camera', fg_color=COLOR_BTN_DANGER, hover_color=COLOR_BTN_DANGER_HOVER)
                self.update_video_loop()
            except RuntimeError as e:
                self.video_label.configure(text=f'Error: {e}')
        else:
            self.stop_camera()

    def stop_camera(self):
        self.is_camera_running = False
        if self.env:
            self.env.close()
            self.env = None
        self.btn_camera.configure(text='Start Camera', fg_color=COLOR_BTN_PRIMARY, hover_color=COLOR_BTN_PRIMARY_HOVER)
        self.video_label.configure(image=None, text="Camera Stopped\nClick 'Start Camera' in the sidebar to resume.")

    def update_video_loop(self):
        if not self.is_camera_running or not self.env:
            return
        ret, frame = self.env.read()
        if ret and frame is not None:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            features = None
            display_frame = frame.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(display_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    raw_coords = HandSample._extract_from_mediapipe(hand_landmarks)
                    sample = HandSample(raw_coords)
                    features = sample.tovector()
            if self.switch_mirror.get():
                display_frame = cv2.flip(display_frame, 1)
            smoothed_sign, confidence = self.agent.act(features)
            locked = self.wb.update(smoothed_sign)
            self._update_hud(smoothed_sign, confidence, locked)
            rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_display)
            ctk_image = ctk.CTkImage(light_image=img, dark_image=img, size=(640, 440))
            self.video_label.configure(image=ctk_image, text='')
            self.video_label.image = ctk_image
        self.after(15, self.update_video_loop)

    def _update_hud(self, sign, conf, locked):
        if sign == 'no sign':
            self.lbl_detected_sign.configure(text='—', text_color=COLOR_STATUS_IDLE)
            self.lbl_confidence.configure(text='Status: No Hand', text_color=COLOR_SUBTITLE)
        elif sign == 'unsure':
            self.lbl_detected_sign.configure(text='—', text_color=COLOR_STATUS_IDLE)
            self.lbl_confidence.configure(text=f'Status: Unsure ({conf * 100:.0f}%)', text_color=COLOR_STATUS_HOLDING)
        else:
            urdu_char = URDU_CHAR_MAP.get(sign, '')
            self.lbl_detected_sign.configure(text=f'{sign.upper()}  {urdu_char}', text_color=COLOR_SIGN_MAIN)
            self.lbl_confidence.configure(text=f'Confidence: {conf * 100:.0f}%', text_color=COLOR_CONFIDENCE)
        progress = min(self.wb.hold_counter / self.wb.hold_frames, 1.0)
        self.progress_bar.set(progress)
        if self.wb.hold_counter == 0:
            self.lbl_hold_status.configure(text='Processing...', text_color=COLOR_STATUS_IDLE)
        elif progress >= 1.0:
            self.lbl_hold_status.configure(text='Locked In!', text_color=COLOR_STATUS_LOCKED)
        else:
            self.lbl_hold_status.configure(text=f'Holding: {self.wb.hold_counter}/{self.wb.hold_frames}', text_color=COLOR_STATUS_HOLDING)
        roman_w = self.wb.get_word()
        urdu_w = self.wb.get_urdu_word()
        self.lbl_roman_word.configure(text=f"Roman: {(roman_w if roman_w else '(empty)')}", text_color=COLOR_WORD_ROMAN)
        self.lbl_urdu_word.configure(text=f"{(urdu_w if urdu_w else '(خالی)')}", text_color=COLOR_WORD_URDU)

    def clear_word(self):
        if self.wb:
            self.wb.clear()
            self.lbl_roman_word.configure(text='Roman: (empty)')
            self.lbl_urdu_word.configure(text='(خالی)')

    def change_theme(self, new_theme):
        ctk.set_appearance_mode(new_theme)

    def on_close(self):
        self.stop_camera()
        self.destroy()

def _extract_from_mediapipe(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords
HandSample._extract_from_mediapipe = staticmethod(_extract_from_mediapipe)

def launch_gui():
    app = PSLApp()
    app.mainloop()
if __name__ == '__main__':
    launch_gui()
