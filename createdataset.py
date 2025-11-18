import cv2
import os

# 📂 Create main dataset directory if it doesn’t exist
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# ✋ Define gestures from A to J
gestures = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' ,'K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# 📸 Set number of images to capture per gesture
dataset_size = 100  # You can increase for better accuracy

cap = cv2.VideoCapture(0)

for gesture in gestures:
    gesture_dir = os.path.join(DATA_DIR, gesture)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f"\n📸 Collecting data for gesture '{gesture}'")
    print("➡️ Press 'c' to capture an image, 'q' to move to next gesture, or 'Esc' to exit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Display current gesture name on screen
        cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Capture Gestures (Press c to Capture, q to Next)', frame)
        key = cv2.waitKey(1)

        if key == ord('c'):
            image_path = os.path.join(gesture_dir, f'{len(os.listdir(gesture_dir))}.jpg')
            cv2.imwrite(image_path, frame)
            print(f"✅ Saved: {image_path}")

            if len(os.listdir(gesture_dir)) >= dataset_size:
                print(f"✅ Finished collecting {dataset_size} images for '{gesture}'")
                break

        elif key == ord('q'):
            print(f"➡️ Moving to next gesture '{gesture}'")
            break

        elif key == 27:  # ESC key
            print("🛑 Exiting early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("\n🎉 Data collection completed successfully for gestures A–Z!")
