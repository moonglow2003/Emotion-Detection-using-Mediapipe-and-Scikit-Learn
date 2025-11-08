import cv2
import mediapipe as mp
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from tqdm import tqdm

# Dataset path
DATA_DIR = "KDEF_Dataset_Front"

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# Labels
emotions = sorted(os.listdir(DATA_DIR))

X = []
y = []

def extract_landmarks(image_path):
    """Extract face landmarks (x,y) and flatten into a vector."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return None

    landmarks = result.multi_face_landmarks[0].landmark
    data = np.array([[lm.x, lm.y] for lm in landmarks]).flatten()
    return data


print("📥 Loading and extracting features...")

for emotion in emotions:
    folder = os.path.join(DATA_DIR, emotion)
    if not os.path.isdir(folder):
        continue

    for img_file in tqdm(os.listdir(folder), desc=emotion):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(folder, img_file)
        landmarks = extract_landmarks(path)
        if landmarks is not None:
            X.append(landmarks)
            y.append(emotion)

X = np.array(X)
y = np.array(y)

print(f"\n✅ Feature extraction done! Total samples: {len(X)}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("🚀 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/emotion_model.pkl")
print("\n💾 Model saved at: models/emotion_model.pkl")
