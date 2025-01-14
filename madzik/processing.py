import keras
from keras import layers
import tensorflow as tf
import numpy as np


class PLUMER:
    def __init__(self, opts):
        self.opts = opts
        self._create_model()

    def _create_model(self):
        self.model = keras.Sequential()
        # Update input shape to 17 channels
        self.model.add(layers.Input(shape=(512, 512, 16)))
        self.model.add(layers.Conv2D(
            32, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10))
        self.model.add(layers.Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.BinaryCrossentropy(),
                           metrics=['accuracy'])
        return self.model

    def preprocess(self, img: np.ndarray):
        return img

    def load_weights(self, path):
        self.model.load_weights(path)

    def parse_image(self, img: np.ndarray):
        self.model.predict(self.preprocess(img))
        return img


plum = PLUMER("opts")
print("PLUMER loaded")
