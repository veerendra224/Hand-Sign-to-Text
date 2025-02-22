import mediapipe as mp
import pickle
import os
import cv2

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Data directory
DATA_DIR = './data'

data = []
labels = []

# Iterate through each directory in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    
    # Skip if it's not a directory
    if not os.path.isdir(dir_path):
        print(f"Skipping non-directory: {dir_path}")
        continue

    # Iterate through each image in the directory
    for img_path in os.listdir(dir_path):
        img_full_path = os.path.join(dir_path, img_path)
        
        # Skip if it's not a file (e.g., subdirectories)
        if not os.path.isfile(img_full_path):
            print(f"Skipping non-file: {img_full_path}")
            continue

        img = cv2.imread(img_full_path)
        if img is None:
            print(f"Warning: Unable to read image {img_full_path}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x)
                    data_aux.append(landmark.y)
            data.append(data_aux)
            labels.append(dir_)

# Save data to pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data processing complete. Saved to data.pickle.")