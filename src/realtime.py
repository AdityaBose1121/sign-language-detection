import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque, Counter

# Load trained model
model = tf.keras.models.load_model("model/sign_model_v3.h5")

labels = ["A", "B", "C", "D", "E"]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Prediction smoothing
pred_queue = deque(maxlen=15)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            xmin, xmax = int(min(x_list) * w), int(max(x_list) * w)
            ymin, ymax = int(min(y_list) * h), int(max(y_list) * h)

            pad = 100
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(w, xmax + pad)
            ymax = min(h, ymax + pad)

            box_w = xmax - xmin
            box_h = ymax - ymin
            size = max(box_w, box_h)

            cx = xmin + box_w // 2
            cy = ymin + box_h // 2

            xmin = max(0, cx - size // 2)
            xmax = min(w, cx + size // 2)
            ymin = max(0, cy - size // 2)
            ymax = min(h, cy + size // 2)

            hand_img = frame[ymin:ymax, xmin:xmax]


            if hand_img.size != 0:
                img = cv2.resize(hand_img, (96, 96))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                pred = model.predict(img, verbose=0)

                confidence = np.max(pred)
                pred_label = labels[np.argmax(pred)]

                print(pred_label, confidence)

                if confidence > 0.75:
                    label = pred_label
                else:
                    label = "..."



                pred_queue.append(label)
                stable_label = Counter(pred_queue).most_common(1)[0][0]

                cv2.putText(
                    frame,
                    stable_label,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3
                )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
