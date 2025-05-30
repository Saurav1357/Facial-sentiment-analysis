import os
import numpy as np
import cv2
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

emotion_labels = {
    'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
    'sad': 4, 'surprise': 5, 'neutral': 6
}

def load_images(folder):
    faces, emotions = [], []
    for emotion in emotion_labels:
        path = os.path.join(folder, emotion)
        if not os.path.isdir(path):
            continue
        for file in os.listdir(path):
            img_path = os.path.join(path, file)
            img = cv2.imread(img_path)
            if img is not None:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (48, 48))
                faces.append(resized)
                emotions.append(emotion_labels[emotion])
    return np.array(faces), np.array(emotions)

print("Loading training data...")
X_train, y_train = load_images('train')
print("Loading testing data...")
X_test, y_test = load_images('test')

X_train = X_train / 255.0
X_test = X_test / 255.0
X_train = X_train.reshape(-1, 48, 48, 1)
X_test = X_test.reshape(-1, 48, 48, 1)

y_train = to_categorical(y_train, 7)
y_test = to_categorical(y_test, 7)


np.savez_compressed("emotion_data_from_images.npz", 
                    X_train=X_train, X_test=X_test,
                    y_train=y_train, y_test=y_test)

print(" Data saved as emotion_data_from_images.npz")