from Assignment1.DNN import *
import numpy as np


def load_data():
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    y_new = np.zeros((10, y_train.shape[0]))
    cnt = 0
    for label in y_train:
        y_new[label][cnt] = 1
        cnt += 1

    y_train = y_new

    y_new = np.zeros((10, y_test.shape[0]))
    cnt = 0
    for label in y_test:
        y_new[label][cnt] = 1
        cnt += 1

    y_test = y_new
    # normalize
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.

    return (x_train, y_train), (x_test, y_test)

split_prob = 0.8
(x_train, y_train), (x_test, y_test) = load_data()

# Split the training data for train-validation set
# validation_split = np.random.rand(len(x_train)) < split_prob
#
# x_train_data = x_train[validation_split]
# x_val_data = x_train[~validation_split]
# y_train_data = y_train[:, validation_split]
# y_val_data = y_train[:, ~validation_split]
#
# x_train_data = x_train_data.reshape((x_train_data.shape[0], 784))
# x_val_data = x_val_data.reshape((x_val_data.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))
layers_dims = [784, 20, 7, 5, 10]
batch_size = 64

params, costs, _, _, _ = L_layer_model(x_train,
                              y_train,
                              layers_dims,
                              learning_rate=0.1,
                              num_iterations=1000,
                              batch_size=64,
                              use_batchnorm=True)

x_train = x_train.reshape((x_train.shape[0], 784))
train_acc = Predict(x_train.T, y_train, params) * 100
test_acc = Predict(x_test.T, y_test, params) * 100

print('Train Accuracy {}%\nTest Accuracy: {}%'.format(train_acc, test_acc))

