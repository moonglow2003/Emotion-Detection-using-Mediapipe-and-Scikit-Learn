import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("models/emotion_model.pkl")

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Emotion labels (same as training)
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Start webcam
cap = cv2.VideoCapture(0)
print("🎥 Webcam started — press 'q' to quit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("❌ Failed to capture frame.")
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for landmarks in result.multi_face_landmarks:
            # Extract features
            data = np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten().reshape(1, -1)

            # Predict emotion
            prediction = model.predict(data)[0]
            emotion_text = prediction.upper()

            # Get face bounding box
            h, w, _ = frame.shape
            xs = [lm.x * w for lm in landmarks.landmark]
            ys = [lm.y * h for lm in landmarks.landmark]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))

            # Draw only bounding box + emotion label
            cv2.rectangle(frame, (x_min, y_min - 30), (x_max, y_min), (0, 255, 0), -1)
            cv2.putText(frame, emotion_text, (x_min + 10, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program ended")

