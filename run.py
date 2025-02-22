import mediapipe as mp
import pickle
import cv2
import numpy as np

# Load the trained model and label encoder
with open('data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = data_dict['data']
labels = data_dict['labels']

# Debugging: Print the shape and content of data and labels
print(f"Number of samples: {len(data)}")
print(f"Number of labels: {len(labels)}")
print(f"Sample data: {data[0] if data else 'No data'}")
print(f"Sample label: {labels[0] if labels else 'No labels'}")

# Convert data and labels to numpy arrays
X = np.array(data)
y = np.array(labels)

# Check if data is empty
if len(X) == 0:
    raise ValueError("No data found in 'data.pickle'. Please ensure the data is collected correctly.")

# Reshape X to 2D if necessary
if X.ndim == 1:
    X = X.reshape(-1, 1)  # Reshape to (n_samples, 1) if X is 1D

# Debugging: Print the shape of X and y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

# Load a simple classifier (e.g., LogisticRegression)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X, y)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from webcam.")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract normalized landmark coordinates
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)

            # Predict the letter
            prediction = model.predict([data_aux])
            predicted_letter = prediction[0]

            # Draw the predicted letter on the frame
            cv2.putText(frame, f"Predicted: {predicted_letter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Sign Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()