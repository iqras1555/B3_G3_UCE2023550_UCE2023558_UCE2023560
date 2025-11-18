# import pickle
# import cv2
# import mediapipe as mp
# import numpy as np

# # ---------- Load Trained Model ----------
# print("🔹 Loading trained model...")
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']
# print("✅ Model loaded successfully!\n")

# # ---------- Initialize MediaPipe Hands ----------
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# )

# # ---------- Open Webcam ----------
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("❌ Error: Could not open webcam. Try changing camera index.")
#     exit()

# print("🎥 Webcam started! Show your gesture (A–J). Press 'ESC' to exit.\n")

# # ---------- Label Dictionary ----------
# labels_dict = {i: chr(65 + i) for i in range(10)}  # 0–9 → A–J

# # ---------- Main Loop ----------
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("⚠️ Skipping empty frame...")
#         continue

#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)

#     data_aux = []
#     x_ = []
#     y_ = []

#     # ---------- If Hands Detected ----------
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
#             )

#             for lm in hand_landmarks.landmark:
#                 x_.append(lm.x)
#                 y_.append(lm.y)

#             for lm in hand_landmarks.landmark:
#                 data_aux.append(lm.x - min(x_))
#                 data_aux.append(lm.y - min(y_))

#         if len(data_aux) == 42:  # 21 landmarks * 2 (x,y)
#             data_aux = np.asarray(data_aux).reshape(1, -1)
#             prediction = model.predict(data_aux)
#             predicted_character = prediction[0]

#             # Confidence (probability of prediction)
#             prediction_proba = model.predict_proba(data_aux)
#             confidence = np.max(prediction_proba) * 100

#             # ---------- Display on Frame ----------
#             cv2.putText(
#                 frame,
#                 f'{predicted_character} ({confidence:.2f}%)',
#                 (10, 50),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2
#             )

#             # ---------- Print in Console ----------
#             print(f"👉 Predicted Gesture: {predicted_character} ({confidence:.2f}%)")

#     cv2.imshow('Hand Gesture Recognition (A–J)', frame)

#     key = cv2.waitKey(1)
#     if key == 27:  # ESC key
#         print("\n🛑 Exiting...")
#         break

# # ---------- Cleanup ----------
# cap.release()
# cv2.destroyAllWindows()

import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from playsound import playsound
import threading
import time
import os

# Load model
print("🔹 Loading model...")
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
print("✅ Model loaded!\n")

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, min_tracking_confidence=0.3)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Cannot open webcam")
    exit()

# Sequence buffer for smoothing
SEQ_LENGTH = 10
sequence_buffer = deque(maxlen=SEQ_LENGTH)
predicted_gesture = "None"
last_gesture = ""
last_time = 0
cooldown = 1.0

sequence_text = ""

def play_sound(letter):
    if not letter:
        return
    sound_file = f"sounds/{letter.upper()}.mp3"
    if os.path.exists(sound_file):
        threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

print("🎥 Webcam started! Press ESC to exit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    overlay_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_, y_, data_aux = [], [], []

            for lm in hand_landmarks.landmark:
                x_.append(lm.x)
                y_.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                new_gesture = prediction[0]

                sequence_buffer.append(new_gesture)
                stable_gesture = max(set(sequence_buffer), key=sequence_buffer.count)

                current_time = time.time()
                if stable_gesture != last_gesture and (current_time - last_time) > cooldown:
                    predicted_gesture = stable_gesture
                    last_gesture = stable_gesture
                    last_time = current_time

                    sequence_text += predicted_gesture
                    play_sound(predicted_gesture)

                overlay_text = f'Gesture: {predicted_gesture}'

    cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Sequence: {sequence_text[-80:]}', (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    cv2.imshow('Hand Gesture Recognition', frame)
    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
