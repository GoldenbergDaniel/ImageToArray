import os
import random

from cv2 import cv2
import numpy as np

import config

training_data = []

for category in config.CATEGORIES:
    category_path = os.path.join(config.DATA_DIR, category)
    for image_path in os.listdir(category_path):
        try:
            img_path = os.path.join(category_path, image_path)
            img_array = cv2.resize(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), (config.SIZE, config.SIZE))
            class_index = config.CATEGORIES.index(category)
            training_data.append([img_array, class_index])
        except:
            pass

random.shuffle(training_data)

images = []
labels = []

for image, label in training_data:
    images.append(image)
    labels.append(label)

percent = 80

np.save(os.path.join(config.SAVES_DIR, "train_images.npy"), images[int(len(images)/percent):])
np.save(os.path.join(config.SAVES_DIR, "train_labels.npy"), labels[int(len(labels)/percent):])
np.save(os.path.join(config.SAVES_DIR, "test_images.npy"), images[:int(len(images)/percent)])
np.save(os.path.join(config.SAVES_DIR, "test_labels.npy"), labels[:int(len(labels)/percent)])
