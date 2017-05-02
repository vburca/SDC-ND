import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, \
    Dropout, MaxPooling2D, Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow

from data_loader import train_samples, validation_samples, \
    train_generator, validation_generator, BATCH_SIZE
from preprocessing import convert_grayscale, normalize

def rgb2grayscale_lambda(x):
    import tensorflow as tf
    return tf.image.rgb_to_grayscale(x)

# X_train, y_train = next(train_generator)
# X_valid, y_valid = next(validation_generator)

model = Sequential()
# TODO I don't like hardcoding stuff like this...
model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(rgb2grayscale_lambda))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))

model.add(Conv2D(32, 2, 2, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.5))
# model.add(Conv2D(64, 2, 2, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(rate=0.5))
model.add(Conv2D(128, 2, 2, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(rate=0.5))

model.add(Flatten())
#model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)
modelcheckpoint_callback = ModelCheckpoint('conv1_model_chk.h5', save_best_only=True)

model.fit_generator( \
    generator=train_generator, \
    steps_per_epoch=len(train_samples) / BATCH_SIZE, \
    validation_data=validation_generator, \
    validation_steps=len(validation_samples) / BATCH_SIZE, \
    epochs=20, \
    callbacks=[earlystop_callback, modelcheckpoint_callback])

model.save('conv1_model.h5')