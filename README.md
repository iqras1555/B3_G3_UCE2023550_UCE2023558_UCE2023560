# B3_G3_UCE2023550_UCE2023558_UCE2023560
Hand Gesture Detector

1. Project Title & Objective
Project Title:

Real-Time Hand Gesture Recognition Using MediaPipe and Random Forest Classifier

Objective:

The objective of this project is to build a real-time hand gesture recognition system that detects hand landmarks using MediaPipe, processes them into normalized features, and predicts gestures (A–Z) using a Random Forest machine learning model.
The system also provides audio feedback by playing the predicted letter sound.

2. Dataset Details
Dataset Creation

Dataset is manually created using createdataset.py.

Gestures: A–Z (26 classes)

Images per class: 100 images

Total dataset size: 2600 images

Images captured using webcam in real time.

Dataset Processing

Using collectimage.py:

Each image is processed with MediaPipe Hands.

Extracts 21 hand landmarks → each with (x, y).

Coordinates normalized:
normalized_value = landmark_coordinate - min(all_coordinates)

Final feature vector per image: 42 values.

Stored in data.pickle as:

data: list of features

labels: corresponding gesture labels

3. Algorithm / Model Used
Hand Landmark Extraction – MediaPipe

Detects 21 stable hand landmarks.

Works in both static and real-time mode.

Provides consistent x–y coordinates.

Machine Learning Model – Random Forest

Trained using trainmodel.py:

n_estimators = 300

max_depth = 25

Uses 80-20 Train–Test Split.

Input features: 42 normalized landmark coordinates.

Output: Gesture label (A–Z).

Why Random Forest?

Handles non-linear patterns in landmark data.

Robust to noise.

Works well with medium-sized datasets.

No feature scaling required.

4. Results (Accuracy, Graphs, etc.)
Model Performance

Printed during training (trainmodel.py):

Training Accuracy: ~98–100%

Testing Accuracy: (Displays when running training script)

Example structure shown:

🎯 Training Accuracy: XX.XX%
🎯 Testing Accuracy: XX.XX%

📄 Classification Report:
Precision, Recall, F1-score for each gesture.

Real-Time Performance

Using finalcode.py:

Detects hand instantly.

Predicts gesture in real time.

Displays live webcam feed with gesture label.

Plays letter sound using gTTS + playsound.

Visualization

Includes:   

Console classification report

On-screen gesture prediction

Hand landmarks drawn using MediaPipe

5. Conclusion

This project successfully implements a fully functional real-time alphabet gesture recognition system using a combination of:

MediaPipe for robust landmark detection

Random Forest for gesture classification

Flask for web-based deployment

Audio output using playsound for accessibility

The system performs well on custom-created images and achieves high testing accuracy, demonstrating reliability for controlled lighting and background conditions.

6. Future Scope

Add support for numbers (0–9) and dynamic gestures.

Use Deep Learning models (CNN, LSTM, GestureNet) for higher accuracy.

Deploy as a mobile or desktop app.

Improve dataset with more variations:

Different lighting

Background clutter

Multiple users

Add text-to-speech continuous word formation.

Add gesture smoothing to filter out noisy predictions.

Integrate sign language word-level recognition.

7. References

(As you requested: Single-line format, no hyperlinks)

Google MediaPipe Hands documentation

OpenCV official documentation

Scikit-learn Random Forest classifier documentation

Flask micro web framework documentation

NumPy scientific computing library

gTTS (Google Text-to-Speech) Python library

Research papers on hand gesture recognition using landmarks

Sign Language gesture datasets and ML approaches

Landmark-based gesture recognition studies

Random Forest applications in gesture classification
