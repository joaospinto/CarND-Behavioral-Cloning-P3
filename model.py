import os
import random

os.environ["KERAS_BACKEND"] = "tensorflow"

with_gpu = True # not massive difference in speed, but faster with GPU
if not with_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import AveragePooling2D, MaxPooling2D

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

tf.app.flags.DEFINE_string(
    'model_file',
    'model.h5',
    "file where model should be saved"
)

tf.app.flags.DEFINE_float(
    'correction',
    0.025, # previously 0.015
    "correction to be applied to center-steering on left and right images"
) #TODO: calibrate further

tf.app.flags.DEFINE_bool(
    'include_sides',
    True,
    "determines whether the images taken from left/right sides are used"
)

tf.app.flags.DEFINE_bool(
    'augment_data',
    False,
    "determines whether data augmentation is used"
)

tf.app.flags.DEFINE_integer(
    'epochs',
    3,
    "number of training epochs"
)

tf.app.flags.DEFINE_integer(
    'raw_batch_size',
    64,
    "batches are constructed from this many base pictures (pre-augmentation)"
) # with flipping + side views, augmented b.s. is 6 * FLAGS.raw_batch_size

tf.app.flags.DEFINE_bool(
    'reset',
    True,
    "If False, previous model will be loaded at the start."
)

def get_data():
    print("Loading data in directory {}...".format(FLAGS.data_dir))
    data = []
    with open(os.path.join(FLAGS.data_dir, "driving_log.csv"), 'r') as f:
        reader = csv.reader(f)
        next(reader, None) # skip headers
        for row in reader:
            data.append(row)
    return data

def get_augmentation(images):
    hsvs = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in images])
    for i in range(hsvs.shape[0]):
        hsv = (np.random.randint(low=0.5, high=1.5) * hsvs[i,:,:,2]).round()
        hsv = np.minimum.reduce([hsv, 255*np.ones_like(hsv)]) # avoid overflow
        hsvs[i,:,:,2] = hsv.astype(np.uint8)
    return [cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) for hsv in hsvs]

def generator(data):
    while True:
        random.shuffle(data)
        for first in range(0, len(data), FLAGS.raw_batch_size):
            last = min(first + FLAGS.raw_batch_size, len(data))
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
            if FLAGS.augment_data:
                augmented = get_augmentation(images)
                images = np.concatenate((images, augmented))
                steering_measurements = np.concatenate((
                    steering_measurements,
                    steering_measurements
                ))
            X_batch = np.concatenate((images, np.fliplr(images)))
            y_batch = np.concatenate(
                (steering_measurements, -steering_measurements)
            )
            yield X_batch, y_batch

def commaai_net():
    model = Sequential()
    model.add(Cropping2D(
        cropping=((60, 30), (0, 0)),
        input_shape=(160, 320, 3)
    ))
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    model.add(Conv2D(filters=4,kernel_size=3,strides=(1,2),activation='elu'))
    model.add(Conv2D(filters=4,kernel_size=3,strides=(1,2),activation='elu'))
    model.add(Conv2D(filters=4,kernel_size=5,strides=(2,3),activation='elu'))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

def nvidia_net():
    # https://arxiv.org/pdf/1604.07316.pdf
    model = Sequential()
    model.add(Cropping2D(
        cropping=((60, 30), (0, 0)), # lectures suggest ((50,20),(0,0))
        input_shape=(160, 320, 3)
    )) # shape (70,320,3)
    model.add(Lambda(lambda img: tf.image.resize_images(img, (35, 160))))
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    model.add(Conv2D(
        filters=24,
        kernel_size=5,
        strides=(1,2), # paper uses (2,2) but our input dims are different
        activation='relu',
        padding="valid"
    ))
    model.add(Conv2D(
        filters=36,
        kernel_size=5,
        strides=(2,2),
        activation='relu',
        padding="valid"
    ))
    model.add(Conv2D(
        filters=48,
        kernel_size=3,
        strides=(2,2),
        activation='relu',
        padding="valid"
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=3,
        strides=(2,2),
        activation='relu',
        padding="valid"
    ))
    model.add(Conv2D(
        filters=64,
        kernel_size=1,
        strides=(2,2),
        activation='relu',
        padding="valid"
    ))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    return model

def mod_lenet():
    model = Sequential()
    model.add(Cropping2D(
        cropping=((60, 30), (0, 0)),
        input_shape=(160, 320, 3)
    )) # shape (70,320,3)
    model.add(Lambda(lambda img: tf.image.resize_images(img, (35, 160))))
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    model.add(Conv2D(
        filters=32,
        kernel_size=5,
        strides=(1,2),
        activation='relu',
        padding="valid"
    ))
    model.add(AveragePooling2D())
    model.add(Conv2D(
        filters=32,
        kernel_size=5,
        strides=(1,2),
        activation='relu',
        padding="same"
    ))
    model.add(AveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    return model

def encoder_net():
    model = Sequential()
    model.add(Cropping2D(
        cropping=((60, 30), (0, 0)),
        input_shape=(160, 320, 3)
    )) # shape (70,320,3)
    model.add(Lambda(lambda img: tf.image.resize_images(img, (35, 160))))
    model.add(Lambda(lambda x: x/127.5 - 1.0))
    model.add(Dropout(0.2))
    model.add(Conv2D(
        filters=32,
        kernel_size=5,
        strides=(1,2),
        activation='relu',
        padding="valid"
    ))
    model.add(AveragePooling2D())
    model.add(Conv2D(
        filters=32,
        kernel_size=5,
        strides=(1,2),
        activation='relu',
        padding="same"
    ))
    model.add(AveragePooling2D())
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    return model

def get_model():
    #return commaai_net()
    #return nvidia_net()
    #return mod_lenet()
    return encoder_net()

def main(_):
    data = get_data()
    random.shuffle(data)
    training_data = data[:int(0.8*len(data))]
    validation_data = data[int(0.8*len(data)):]
    train_gen = generator(training_data)
    valid_gen = generator(validation_data)
    model = get_model()
    if not FLAGS.reset:
        model = load_model(FLAGS.model_file)
    print(model.summary())
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=int(np.ceil(len(training_data)/FLAGS.raw_batch_size)),
        epochs=FLAGS.epochs,
        callbacks=[
            ModelCheckpoint(FLAGS.model_file),
            TensorBoard(
                write_images=True,
                batch_size=FLAGS.raw_batch_size
            )
        ],
        verbose=1,
        validation_data=valid_gen,
        validation_steps=int(np.ceil(len(validation_data)/FLAGS.raw_batch_size))
    )

if __name__ == '__main__':
    tf.app.run()
