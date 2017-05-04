import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Cropping2D, Lambda, \
    Dropout, MaxPooling2D, Activation
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

import tensorflow

from data_loader import train_samples, validation_samples, \
    train_generator, validation_generator, BATCH_SIZE

def preprocessing(x):
    def get_saturation(x):
        import tensorflow as tf
        hsv = tf.image.rgb_to_hsv(x)
        return tf.map_fn(fn=lambda i: tf.slice(i, [0, 0, 1], [-1, -1, 1]), elems=hsv)

    def normalize(x):
        return (x / 255.0) - 0.5

    def resize(x):
        import tensorflow as tf
        return tf.image.resize_images(x, [32, 64])

    x = x / 255.
    x = get_saturation(x)
    x = x - 0.5
    x = resize(x)
    return x

model = Sequential()

model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(preprocessing))

model.add(Conv2D(8, (2, 2), activation='relu'))
model.add(MaxPooling2D((4, 4)))
# model.add(Dropout(rate=0.3))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Dropout(rate=0.3))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=2)
modelcheckpoint_callback = ModelCheckpoint('conv8_model_chk.h5', save_best_only=True)

model.fit_generator( \
    generator=train_generator, \
    steps_per_epoch=len(train_samples) * 3 / BATCH_SIZE, \
    validation_data=validation_generator, \
    validation_steps=len(validation_samples) * 3 / BATCH_SIZE, \
    epochs=10, \
    verbose=1, \
    callbacks=[earlystop_callback, modelcheckpoint_callback])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('conv8_model.h5')