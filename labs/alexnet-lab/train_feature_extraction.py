import pickle
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
import time

# Normalization
def normalize(dataset):
    mean = np.mean(dataset)
    stddev = np.std(dataset)

    return (dataset - mean) / stddev


# TODO: Load traffic signs data.
training_file = "train.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)

features, labels = train['features'], train['labels']

# Display summary of training data
n_train = features.shape[0]
n_classes = np.unique(labels).size

print("Number of examples= ", n_train)
print("Number of classes= ", n_classes)

# Normalize the data sets
# features = normalize(features)

# TODO: Split data into training and validation sets.
# First let's shuffle the data, just in case
# flatten_X = features.reshape(len(features), -1)
# flatten_y = labels.reshape(len(labels), -1)
# concat_data = np.c_[flatten_X, flatten_y]

# # Get views of the concatenated version
# X_view = concat_data[:, :features.size//len(features)].reshape(features.shape)
# y_view = concat_data[:, features.size//len(features):].reshape(labels.shape)

# # Now shuffle the concatenated version, which will properly
# # keep the views in sync in terms of shuffling and indexing
# np.random.shuffle(concat_data)

# # Now select a percentage of data for training, remaining for validation
# training_percentage = .8 # 80%
# train_last_index = int(features.shape[0] * training_percentage)

# X_train = features[:train_last_index]
# y_train = labels[:train_last_index]

# X_validation = features[train_last_index:]
# y_validation = labels[train_last_index:]

X_train, X_validation, y_train, y_validation = train_test_split(features, labels, test_size=0.33, random_state=0)

print("Number of training examples= ", X_train.shape[0])
print("Number of validation examples= ", X_validation.shape[0])
# print("Number of training classes= ", y_train.shape[0])
# print("Number of validation classes= ", y_validation.shape[0])

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, features.shape[1], features.shape[2], features.shape[3]))  # these should be 32, 32, 3
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
# Get the dimensions of the output of the previous layer
prev_shape = fc7.get_shape().as_list()[-1]

# Create the weights and biases for the final fully connected layer
fc8W = tf.Variable(tf.truncated_normal((prev_shape, n_classes), mean=0.0, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(n_classes))

# Do the multiplication xW + b
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)  # or tf.matmul(fc7, fc8W) + fc8b
# probabilities = tf.nn.softmax(logits)  # we might not need this, for efficiency of using tf.nn.softmax_cross_entropy_with_logits

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
one_hot_y = tf.one_hot(y, n_classes)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
training_operation = optimizer.minimize(loss_operation, var_list=[fc8W, fc8b])  # only minimize the loss on the weights and biases of the last layer

# TODO: Train and evaluate the feature extraction model.
# Set the epochs number and batch size
EPOCHS = 10
BATCH_SIZE = 128

# Define the evaluation of the model
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        end = offset + BATCH_SIZE
        batch_x, batch_y = X_data[offset:end], y_data[offset:end]
        accuracy, loss = sess.run([accuracy_operation, loss_operation], feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += (loss * len(batch_x))

    return (total_accuracy / num_examples), (total_loss / num_examples)

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)

    print("Training...")
    print()

    for i in range(EPOCHS):
        # Start timer
        start_time = time.time()

        # For each epoch, shuffle
        X_train, y_train = shuffle(X_train, y_train)

        # Get batches
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]

            # Run model on batch
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        # Print details
        validation_accuracy, validation_loss = evaluate(X_validation, y_validation)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = %.3f" % validation_accuracy)
        print("Validation loss = %.3f" % validation_loss)
        print("Time to train epoch = %.3f seconds" % (time.time() - start_time))
        print()

    saver.save(sess, './alexnet-traffic-signs-classifier')
    print("Model saved")
