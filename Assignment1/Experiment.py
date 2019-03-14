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
    # normalize
    x_train = x_train.astype(float) / 255.
    x_test = x_test.astype(float) / 255.

    return (x_train, y_train), (x_test, y_test)


def run_Experiment(x_train, y_train, layers_dim, x_val_data, y_val_data, x_test, y_test, batch_size, use_batchnorm):
    num_of_epoches = 0
    no_improve_count = 0
    curr_acc = 0.0
    costs_per_epoch = []
    train_flag = True
    while (train_flag):
        num_of_epoches += 1
        params, costs = L_layer_model(x_train.T, y_train, layers_dim, batch_size=batch_size,use_batchnorm=use_batchnorm)
        costs_per_epoch.append(costs)
        acc = Predict(x_val_data, y_val_data, params)
        if acc < curr_acc:
            no_improve_count += 1
        else:
            curr_acc = acc
            no_improve_count = 0
        if no_improve_count == 100:
            train_flag = False
    val_acc = acc
    train_acc = Predict(x_train, y_train, params)
    test_acc = Predict(x_test, y_test, params)
    return num_of_epoches, val_acc, train_acc, test_acc, batch_size, costs_per_epoch


split_prob = 0.8
(x_train, y_train), (x_test, y_test) = load_data()



# Split the training data for train-validation set
validation_split = np.random.rand(len(x_train)) < split_prob

x_train_data = x_train[validation_split]
x_val_data = x_train[~validation_split]
y_train_data = y_train[:,validation_split]
y_val_data = y_train[:,~validation_split]

x_train_data = x_train_data.reshape((x_train_data.shape[0],784))
layers_dims = [x_train_data.shape[1], 20, 7, 5, 10]
batch_size = 64

num_of_epochs, val_acc, train_acc, test_acc, batch_size_new, costs_per_epoch = run_Experiment(x_train_data, y_train_data,
                                                                                               layers_dims, x_val_data,
                                                                                               y_val_data, x_test,
                                                                                               y_test, batch_size,
                                                                                               False)
# num_of_epoches, val_acc, train_acc, test_acc, batch_size_new, costs_per_epoch = run_Experiment(x_train, y_train,
#                                                                                                layers_dims, x_val_data,
#                                                                                                y_val_data, x_test,
#                                                                                                y_test, batch_size, True)
