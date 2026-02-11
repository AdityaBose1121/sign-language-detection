import cv2
import mediapipe as mp
import numpy as np
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True)

dataset_path = "dataset"
X = []
y = []

labels = sorted(os.listdir(dataset_path))

count = 0

for label_index, label in enumerate(labels):
    folder = os.path.join(dataset_path, label)

    for file in os.listdir(folder):
        img_path = os.path.join(folder, file)
        image = cv2.imread(img_path)

        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0]

            data = []
            for lm in landmarks.landmark:
                data.extend([lm.x, lm.y, lm.z])

            X.append(data)
            y.append(label_index)

            count += 1

            if count % 500 == 0:
                print("Processed:", count)


print("Samples collected:", len(X))

np.save("X.npy", np.array(X))
np.save("y.npy", np.array(y))
