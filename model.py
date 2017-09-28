import os
import random

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf

from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential, Model # VGG19 uses functional API
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import csv
import cv2

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'data_dir',
    'data/',
    "Directory containing driving_log.csv and IMG/*.jpg"
)

def get_data():
    data = []
    with open(os.path.join(FLAGS.data_dir, "driving_log.csv"), 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip headers
        for row in reader:
            data.append(row)
    return data

def generator(data, half_batch_size=64): # images are flipped
    while True:
        random.shuffle(data)
        for first in range(0, len(data), half_batch_size):
            last = min(first + half_batch_size, len(data))
            center_images = []
            steering_measurements = []
            for c, l, r, steering, throttle, brake, speed in data[first:last]:
                img = cv2.imread(os.path.join(FLAGS.data_dir, c))
                center_images.append(img)
                steering_measurements.append(steering)
            center_images = np.array(center_images)
            steering_measurements = np.array(
                steering_measurements, dtype=np.float32
            )
            X_batch = np.concatenate((center_images, np.fliplr(center_images)))
            y_batch = np.concatenate(
                (steering_measurements, -steering_measurements)
            )
            yield X_batch, y_batch

def inception():
    inception = InceptionV3(
        weights='imagenet',
        include_top=False,
        input_shape=(160, 320, 3)
    )
    for layer in inception.layers:
        layer.trainable = False
    x = inception.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="selu")(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation="selu")(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation="selu")(x)
    x = Dropout(0.5)(x)
    steering = Dense(1)(x) # -1.0 <= steering <= 1.0
    model = Model(inputs=inception.input, outputs=steering)
    return model

def lenet():
    model = Sequential()
    model.add(Conv2D(6, 5, activation='relu', input_shape=(160, 320, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(16, 5, activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))
    return model

def main(_):
    #TODO:
    #1: add cropping
    #2: deal with bottleneck separately
    data = get_data()
    random.shuffle(data)
    training_data = data[:int(0.8*len(data))]
    validation_data = data[int(0.8*len(data)):]
    half_batch_size=64
    train_gen = generator(training_data, half_batch_size)
    valid_gen = generator(validation_data, half_batch_size)
    model = inception()
    #model = lenet()
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(len(training_data)/half_batch_size)),
        epochs=5,
        callbacks=[],
        validation_data=valid_gen,
        validation_steps=int(np.ceil(len(validation_data)/half_batch_size)),
        verbose=1
    )
    # Now that we have trained the final layers, we fine-tune the whole net.
    for layer in model.layers:
        layer.trainable = True
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(len(training_data)/half_batch_size)),
        epochs=5,
        callbacks=[],
        validation_data=valid_gen,
        validation_steps=int(np.ceil(len(validation_data)/half_batch_size)),
        verbose=1
    )
    model.save('model.h5', overwrite=True)

if __name__ == '__main__':
    tf.app.run()
