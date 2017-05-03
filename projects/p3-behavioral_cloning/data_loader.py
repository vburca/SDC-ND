import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import random

data_sources = [
    'data',
    'my-data',
    'my-keys-data',
    'counterclock-keys-data'
]

BATCH_SIZE = 128
DEVIATION_DELTA = 0.10

""" Method to load data lines from a data source that contains a .csv file.
    Returns data lines from the .csv file.
"""
def _load_data_source(data_source):
    lines = []
    with open('data-sources/' + data_source + '/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            # Skip the header line, if we have one
            if 'center' in line:
                continue
            lines.append(line)
    return lines

""" Method to load all data lines.
    Returns data lines from all the data .csv files.
"""
def _load_all_data_sources():
    lines = []
    for data_source in data_sources:
        lines += _load_data_source(data_source)

    return lines

def _get_center_image_steering(batch_sample):
    center_image_path = batch_sample[0].strip()
    center_image = cv2.imread(center_image_path)
    center_angle = float(batch_sample[3])
    return center_image, center_angle

def _get_left_image_steering(batch_sample):
    left_image_path = batch_sample[1].strip()
    left_image = cv2.imread(left_image_path)
    left_angle = float(batch_sample[3]) + DEVIATION_DELTA
    return left_image, left_angle

def _get_right_image_steering(batch_sample):
    right_image_path = batch_sample[2].strip()
    right_image = cv2.imread(right_image_path)
    right_angle = float(batch_sample[3]) - DEVIATION_DELTA
    return right_image, right_angle

""" Generator for image and steering data, so that we do not load all
the data in memory at once.
"""
def _generator(samples, batch_size=BATCH_SIZE):
    num_samples = len(samples)

    while True:
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                prob = random.random()
                if prob < .33:
                    image, angle = _get_left_image_steering(batch_sample)
                elif prob < .66:
                    image, angle = _get_left_image_steering(batch_sample)
                else:
                    image, angle = _get_right_image_steering(batch_sample)

                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_samples, validation_samples = train_test_split(_load_all_data_sources(), test_size=0.2)

train_generator = _generator(train_samples * 3)
validation_generator = _generator(validation_samples * 3)