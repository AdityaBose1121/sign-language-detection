import cv2
import mediapipe as mp
import numpy as np
import os
from multiprocessing import Pool, cpu_count

dataset_path = "dataset"

mp_hands = mp.solutions.hands


def init_worker():
    global hands
    hands = mp_hands.Hands(static_image_mode=True)


def process_image(args):
    img_path, label_index = args

    image = cv2.imread(img_path)
    if image is None:
        return None

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]

        data = []
        for lm in landmarks.landmark:
            data.extend([lm.x, lm.y, lm.z])

        return data, label_index

    return None


def main():
    labels = sorted(os.listdir(dataset_path))

    tasks = []
    for label_index, label in enumerate(labels):
        folder = os.path.join(dataset_path, label)

        for file in os.listdir(folder):
            img_path = os.path.join(folder, file)
            tasks.append((img_path, label_index))

    print("Total images:", len(tasks))

    # SAFE worker count
    workers = 8
    print("Using", workers, "workers")

    X, y = [], []

    with Pool(workers, initializer=init_worker) as pool:
        for i, result in enumerate(pool.imap_unordered(process_image, tasks)):
            if result:
                data, label_index = result
                X.append(data)
                y.append(label_index)

            if i % 500 == 0 and i > 0:
                print("Processed:", i)

    print("Samples collected:", len(X))

    np.save("X.npy", np.array(X))
    np.save("y.npy", np.array(y))


if __name__ == "__main__":
    main()
