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
    # Convert image to HSV
    def get_saturation(x):
        import tensorflow as tf
        hsv = tf.image.rgb_to_hsv(x)
        return tf.map_fn(fn=lambda i: tf.slice(i, [0, 0, 1], [-1, -1, 1]), elems=hsv)

    # Resize to reduce the memory and computational footprint
    def resize(x):
        import tensorflow as tf
        return tf.image.resize_images(x, [32, 64])

    # We first need to normalize to [0, 1] due to Tensorflow's HSV conversion
    x = x / 255.
    x = get_saturation(x)
    # Now we normalize to [-0.5, 0.5]
    x = x - 0.5
    x = resize(x)
    return x

model = Sequential()

model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(preprocessing))

model.add(Conv2D(16, (2, 2), activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Conv2D(32, (2, 2), activation='relu'))
model.add(MaxPooling2D((4, 4)))
model.add(Dropout(rate=0.2))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer='adam')

# Used for early stop; realized that I actually do want all 10 epochs to train the model
# in order to get best performance.
# earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3)
modelcheckpoint_callback = ModelCheckpoint('test_model_chk.h5', save_best_only=True)

# We multiply the steps by 3 because of the way the generators are constructed -
# given that the test generator randomly picks one of the center, left or right images
# using a probabilistic model, we want to go over the train samples 3 times overall, since
# each training sample contains 3 possible images.
model.fit_generator( \
    generator=train_generator, \
    steps_per_epoch=len(train_samples) * 3 / BATCH_SIZE, \
    validation_data=validation_generator, \
    validation_steps=len(validation_samples) * 3 / BATCH_SIZE, \
    epochs=10, \
    verbose=1, \
    callbacks=[modelcheckpoint_callback])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('test_model.h5')