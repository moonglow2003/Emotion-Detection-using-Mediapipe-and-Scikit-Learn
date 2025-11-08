import cv2
import mediapipe as mp
import os
from tqdm import tqdm

# Input and output dataset paths
INPUT_DIR = "KDEF Dataset"
OUTPUT_DIR = "KDEF_Dataset_Front"

# Make output structure
os.makedirs(OUTPUT_DIR, exist_ok=True)
for emotion in os.listdir(INPUT_DIR):
    emotion_path = os.path.join(INPUT_DIR, emotion)
    if os.path.isdir(emotion_path):
        os.makedirs(os.path.join(OUTPUT_DIR, emotion), exist_ok=True)

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

def is_front_face(landmarks):
    """Check if the face is front-facing based on eye-nose symmetry."""
    LEFT_EYE = 33
    RIGHT_EYE = 263
    NOSE = 1

    lx, ly = landmarks[LEFT_EYE].x, landmarks[LEFT_EYE].y
    rx, ry = landmarks[RIGHT_EYE].x, landmarks[RIGHT_EYE].y
    nx, ny = landmarks[NOSE].x, landmarks[NOSE].y

    left_dist = abs(nx - lx)
    right_dist = abs(rx - nx)
    ratio = left_dist / right_dist if right_dist != 0 else 0

    # Ratio close to 1 => front face
    return 0.8 <= ratio <= 1.25


def process_image(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        return False

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        return False

    landmarks = result.multi_face_landmarks[0].landmark
    if is_front_face(landmarks):
        cv2.imwrite(save_path, img)
        return True
    return False


def main():
    total, kept = 0, 0

    for emotion in os.listdir(INPUT_DIR):
        emotion_dir = os.path.join(INPUT_DIR, emotion)
        if not os.path.isdir(emotion_dir):
            continue

        output_emotion_dir = os.path.join(OUTPUT_DIR, emotion)
        os.makedirs(output_emotion_dir, exist_ok=True)

        print(f"\n🔍 Processing emotion: {emotion}")
        for file in tqdm(os.listdir(emotion_dir)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            total += 1
            img_path = os.path.join(emotion_dir, file)
            save_path = os.path.join(output_emotion_dir, file)

            if process_image(img_path, save_path):
                kept += 1

    print(f"\n✅ Done! Kept {kept}/{total} total images as front faces.")
    print(f"Filtered dataset saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
