import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter
import os

# Load model
model = tf.keras.models.load_model("model/landmark_model.h5")

labels = sorted(os.listdir("dataset"))

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

pred_queue = deque(maxlen=15)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = ""

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        data = []
        handedness = result.multi_handedness[0].classification[0].label

        for lm in hand_landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])


        data = np.array(data).reshape(1, -1)

        pred = model.predict(data, verbose=0)
        pred_label = labels[np.argmax(pred)]

        pred_queue.append(pred_label)
        label = Counter(pred_queue).most_common(1)[0][0]

    cv2.putText(frame, label, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                2, (0,255,0), 3)

    cv2.imshow("Landmark Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
