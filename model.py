import os
import random

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'data/',
    "directory containing driving_log.csv and IMG/*.jpg"
)

tf.app.flags.DEFINE_float(
    'correction',
    0.01,
    "correction to be applied to center-steering on left and right images"
)

tf.app.flags.DEFINE_bool(
    'include_sides',
    False,
    "determines whether the images taken from left/right sides are used"
)

def get_data():
    data = []
    with open(os.path.join(FLAGS.data_dir, "driving_log.csv"), 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip headers
        for row in reader:
            data.append(row)
    return data

def generator(data, half_batch_size=64): # filpped images are also included
    while True:
        random.shuffle(data)
        for first in range(0, len(data), half_batch_size):
            last = min(first + half_batch_size, len(data))
            center_images = []
            left_images = []
            right_images = []
            steering_measurements = []
            for c, l, r, steering, throttle, brake, speed in data[first:last]:
                # strip is used to remove leading whitespace
                img_c = cv2.imread(os.path.join(FLAGS.data_dir, c.strip()))
                img_l = cv2.imread(os.path.join(FLAGS.data_dir, l.strip()))
                img_r = cv2.imread(os.path.join(FLAGS.data_dir, r.strip()))
                center_images.append(img_c)
                left_images.append(img_l)
                right_images.append(img_r)
                steering = float(steering)
                steering_measurements.append(steering)
            images = np.array(center_images)
            steering_measurements = np.array(steering_measurements)
            if FLAGS.include_sides:
                images = np.concatenate((
                    images,
                    np.array(left_images),
                    np.array(right_images)
                ))
                steering_measurements = np.concatenate((
                    steering_measurements,
                    steering_measurements + FLAGS.correction,
                    steering_measurements - FLAGS.correction
                ))
            X_batch = np.concatenate((images, np.fliplr(images)))
            y_batch = np.concatenate(
                (steering_measurements, -steering_measurements)
            )
            yield X_batch, y_batch

def get_model():
    model = Sequential()
    model.add(Cropping2D(
        cropping=((70, 30), (0, 0)),
        input_shape=(160, 320, 3)
    ))
    model.add(Lambda(lambda x: x/255.0 - 0.5))
    model.add(Conv2D(filters=4,kernel_size=3,strides=(1,2),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=4,kernel_size=3,strides=(1,2),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=4,kernel_size=5,strides=(2,3),activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv2D(filters=4,kernel_size=5,strides=(2,2),activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    return model

def main(_):
    data = get_data()
    random.shuffle(data)
    training_data = data[:int(0.8*len(data))]
    validation_data = data[int(0.8*len(data)):]
    half_batch_size=64
    train_gen = generator(training_data, half_batch_size)
    valid_gen = generator(validation_data, half_batch_size)
    model = get_model()
    print(model.summary())
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(len(training_data)/half_batch_size)),
        epochs=3,
        callbacks=[], #TODO
        validation_data=valid_gen,
        validation_steps=int(np.ceil(len(validation_data)/half_batch_size)),
        verbose=1
    )
    model.save_weights('weights.h5', overwrite=True)

if __name__ == '__main__':
    tf.app.run()
