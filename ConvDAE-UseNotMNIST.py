from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import matplotlib.pyplot as plt
import random

sess = tf.Session()

pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset']
    valid_labels = save['valid_labels']
    test_dataset = save['test_dataset']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory

# Reformat into a TensorFlow-friendly shape:
# - convolutions need the image data formatted as a cube (width by height by #channels)
# - labels as float 1-hot encodings.

image_size = 28
num_labels = 10
num_channels = 1  # grayscale


def reformat(dataset, labels):    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

batch_size = 128
num_steps = 3001
num_hiddens = 64

# Input data.
inputs = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))
targets = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channels))

# Model.
conv1 = tf.layers.conv2d(inputs, 64, (5, 5), padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1, (2, 2), (2, 2), padding='same')

conv2 = tf.layers.conv2d(pool1, 64, (5, 5), padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2, (2, 2), (2, 2), padding='same')

conv3_resize = tf.image.resize_nearest_neighbor(pool2, (14, 14))
conv4 = tf.layers.conv2d(conv3_resize, 64, (5, 5), padding='same', activation=tf.nn.relu)

conv4_resize = tf.image.resize_nearest_neighbor(conv4, (28, 28))
conv5 = tf.layers.conv2d(conv4_resize, 64, (5, 5), padding='same', activation=tf.nn.relu)

y_conv = tf.layers.conv2d(conv5, 1, (5, 5), padding='same', activation=None)

outputs = tf.nn.sigmoid(y_conv)

loss = tf.reduce_mean(tf.nn.l2_loss(y_conv - targets))
optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)

noise_factor = 0.5
sess.run(tf.global_variables_initializer())
for step in range(num_steps):
    offset = (step * batch_size) % (train_dataset.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    noisy_data = batch_data + noise_factor * np.random.randn(*batch_data.shape)
    noisy_data = np.clip(noisy_data, 0.0, 1.0)
    _, l = sess.run([optimizer, loss], feed_dict={inputs: noisy_data, targets: batch_data})
    if step % 50 == 0:
        print('Loss at step %d: %f ' % (step, l))

fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(20, 4))
in_imgs = test_dataset[random.sample(range(500), 10)]
noisy_imgs = in_imgs + noise_factor * np.random.randn(*in_imgs.shape)
noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)

reconstructed = sess.run(outputs, feed_dict={inputs: noisy_imgs.reshape((10, 28, 28, 1))})

for images, row in zip([noisy_imgs, reconstructed], axes):
    for img, ax in zip(images, row):
        ax.imshow(img.reshape((28, 28)))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

fig.tight_layout(pad=0.1)

plt.show()

sess.close()
