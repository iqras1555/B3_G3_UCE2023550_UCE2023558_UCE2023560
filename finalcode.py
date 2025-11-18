# from flask import Flask, render_template, Response, jsonify, request
# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle
# import threading
# import time
# from playsound import playsound
# import os

# # ----------------------------
# # Load model
# # ----------------------------
# model_dict = pickle.load(open('./model.p', 'rb'))
# model = model_dict['model']

# # ----------------------------
# # Initialize Flask & Mediapipe
# # ----------------------------
# app = Flask(__name__)
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils

# camera_running = False
# cap = None
# predicted_gesture = "None"
# last_gesture = ""
# last_time = 0

# # ----------------------------
# # Play letter sound instantly
# # ----------------------------
# def play_sound(letter):
#     sound_file = f"sounds/{letter.upper()}.mp3"
#     if os.path.exists(sound_file):
#         threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

# # ----------------------------
# # Generate video frames
# # ----------------------------
# def gen_frames():
#     global cap, camera_running, predicted_gesture, last_gesture, last_time
#     hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#     cooldown = 1.2  # seconds between each sound

#     while camera_running:
#         success, frame = cap.read()
#         if not success:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 x_, y_, data_aux = [], [], []

#                 for lm in hand_landmarks.landmark:
#                     x_.append(lm.x)
#                     y_.append(lm.y)
#                 for lm in hand_landmarks.landmark:
#                     data_aux.append(lm.x - min(x_))
#                     data_aux.append(lm.y - min(y_))

#                 if len(data_aux) == 42:
#                     prediction = model.predict([np.asarray(data_aux)])
#                     new_gesture = prediction[0]
#                     current_time = time.time()

#                     if new_gesture != last_gesture and (current_time - last_time) > cooldown:
#                         predicted_gesture = new_gesture
#                         last_gesture = new_gesture
#                         last_time = current_time

#                         print(f"Predicted Gesture: {predicted_gesture}")
#                         play_sound(predicted_gesture)  # Play instantly

#                         cv2.putText(frame, f'Gesture: {predicted_gesture}', (10, 40),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#         else:
#             cv2.putText(frame, "No Hand Detected", (10, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     hands.close()

# # ----------------------------
# # Routes
# # ----------------------------
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/start_camera', methods=['POST'])
# def start_camera():
#     global cap, camera_running
#     if not camera_running:
#         cap = cv2.VideoCapture(0)
#         camera_running = True
#     return jsonify({"status": "Camera started"})

# @app.route('/stop_camera', methods=['POST'])
# def stop_camera():
#     global cap, camera_running
#     camera_running = False
#     time.sleep(0.5)
#     if cap:
#         cap.release()
#     return jsonify({"status": "Camera stopped"})

# @app.route('/video_feed')
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/get_gesture')
# def get_gesture():
#     global predicted_gesture
#     return jsonify({"gesture": predicted_gesture})

# @app.route('/exit', methods=['POST'])
# def exit_app():
#     global camera_running
#     camera_running = False
#     if cap:
#         cap.release()
#     func = request.environ.get('werkzeug.server.shutdown')
#     if func:
#         func()
#     return jsonify({"status": "Server shutting down..."})

# # ----------------------------
# # Run app
# # ----------------------------
# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, Response, jsonify, request
import cv2
import mediapipe as mp
import numpy as np
import pickle
import threading
import time
from playsound import playsound
import os
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)

# ----------------------------
# Load single-frame model (RandomForest)
# ----------------------------
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
logging.info("Loaded RandomForest model from model.p")

# ----------------------------
# Initialize Flask & Mediapipe
# ----------------------------
app = Flask(__name__)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

camera_running = False
cap = None

# prediction state
predicted_gesture = "None"
last_gesture = ""
last_time = 0

# sequence state (the built sentence / words)
sequence_text = ""
sequence_lock = threading.Lock()

# ----------------------------
# Sequence buffer for smoothing predictions
# ----------------------------
SEQ_LENGTH = 10  # Number of frames to smooth
sequence_buffer = deque(maxlen=SEQ_LENGTH)

# ----------------------------
# Play letter sound instantly (non-blocking)
# ----------------------------
def play_sound(letter):
    if not letter:
        return
    sound_file = f"sounds/{letter.upper()}.mp3"
    if os.path.exists(sound_file):
        threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

# ----------------------------
# Helper: append to sequence safely
# ----------------------------
def append_to_sequence(token):
    global sequence_text
    with sequence_lock:
        if token == "blank" or token == "space":
            if not sequence_text.endswith(" "):
                sequence_text += " "
        else:
            sequence_text += token

# ----------------------------
# Generate video frames
# ----------------------------
def gen_frames():
    global cap, camera_running, predicted_gesture, last_gesture, last_time, sequence_buffer
    hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cooldown = 1.0  # seconds between gestures

    while camera_running:
        success, frame = cap.read()
        if not success:
            break

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
                    # Predict gesture frame-wise
                    prediction = model.predict([np.asarray(data_aux)])
                    new_gesture = prediction[0]

                    # Add to buffer
                    sequence_buffer.append(new_gesture)
                    # Use majority vote for stable gesture
                    stable_gesture = max(set(sequence_buffer), key=sequence_buffer.count)

                    current_time = time.time()
                    if stable_gesture != last_gesture and (current_time - last_time) > cooldown:
                        predicted_gesture = stable_gesture
                        last_gesture = stable_gesture
                        last_time = current_time

                        append_to_sequence(predicted_gesture)
                        play_sound(predicted_gesture)

                    overlay_text = f'Gesture: {predicted_gesture}'
        else:
            overlay_text = "No Hand Detected"

        # Draw overlay text
        cv2.putText(frame, overlay_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        with sequence_lock:
            display_seq = sequence_text[-80:]
        cv2.putText(frame, f'Sequence: {display_seq}', (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    hands.close()

# ----------------------------
# Flask Routes
# ----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap, camera_running
    if not camera_running:
        cap = cv2.VideoCapture(0)
        camera_running = True
    return jsonify({"status": "Camera started"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap, camera_running
    camera_running = False
    time.sleep(0.5)
    if cap:
        cap.release()
    return jsonify({"status": "Camera stopped"})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_gesture')
def get_gesture():
    global predicted_gesture
    return jsonify({"gesture": predicted_gesture})

@app.route('/get_sequence')
def get_sequence():
    with sequence_lock:
        return jsonify({"sequence": sequence_text})

@app.route('/clear_sequence', methods=['POST'])
def clear_sequence():
    global sequence_text
    with sequence_lock:
        sequence_text = ""
    return jsonify({"status": "Sequence cleared"})

@app.route('/exit', methods=['POST'])
def exit_app():
    global camera_running
    camera_running = False
    if cap:
        cap.release()
    func = request.environ.get('werkzeug.server.shutdown')
    if func:
        func()
    return jsonify({"status": "Server shutting down..."})

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
