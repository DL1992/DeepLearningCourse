from Assignment1.DNN import *
import numpy as np
import matplotlib.pyplot as plt

# from keras.datasets import mnist

def load_data():
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')
    # x_train, y_train, x_test, y_test = mnist.load()

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


def plot_multiple_imgs(X, y, nrow=2, ncol=2, figsize=(13, 7), preds=None, skip=0):
    fig, ax = plt.subplots(nrows=nrow, ncols=ncol, figsize=figsize)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    for i in range(nrow * ncol):
        ax[i // ncol, i % ncol].imshow(X[skip + i], cmap='binary')
        ax[i // ncol, i % ncol].set_xticks([])
        ax[i // ncol, i % ncol].set_yticks([])
        if preds is not None:
            ax[i // ncol, i % ncol].text(0.85, 0.1, str(preds[skip + i]), transform=ax[i // ncol, i % ncol].transAxes,
                                         color='green' if y[skip + i] == preds[skip + i] else 'red', weight='bold')
            ax[i // ncol, i % ncol].text(0.05, 0.1, str(y[skip + i]), color='blue',
                                         transform=ax[i // ncol, i % ncol].transAxes, weight='bold')
        else:
            ax[i // ncol, i % ncol].text(0.05, 0.1, str(y[skip + i]), color='blue',
                                         transform=ax[i // ncol, i % ncol].transAxes, weight='bold')
    plt.show()


def plot_learning(costs):
    costl = []
    steps = []
    accs = []
    for cost, step, acc in costs:
        costl.append(cost)
        steps.append(step)
        accs.append(acc)
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.plot(steps, costl, label='cost')
    ax2.plot(steps, accs, label='accuracy')
    ax1.set_xlabel('training step')
    ax1.set_ylabel('cost')
    ax1.set_title('cost over training steps')
    ax1.legend()
    ax2.set_xlabel('training step')
    ax2.set_ylabel('accuracy)')
    ax2.set_title('accuracy over training steps')
    ax2.legend()
    plt.show()


def run_experiment(x, y, layers_dims, val_split_prob=0.8, learning_rate=0.009, num_iterations=300,
                   batch_size=32,
                   use_batchnorm=True):
    params, costs, num_of_epochs, num_of_training_steps, val_acc, num_of_iterations = L_layer_model(x,
                                                                                                    y,
                                                                                                    layers_dims,
                                                                                                    val_split_prob=val_split_prob,
                                                                                                    learning_rate=learning_rate,
                                                                                                    num_iterations=num_iterations,
                                                                                                    batch_size=batch_size,
                                                                                                    use_batchnorm=use_batchnorm)

    x_train_d = x_train.reshape((x_train.shape[0], 784))
    train_acc = Predict(x_train_d.T, y_train, params, use_batchnorm=use_batchnorm) * 100
    test_acc = Predict(x_test.T, y_test, params, use_batchnorm=use_batchnorm) * 100
    plot_learning(costs)
    print(
        f'The ANN ran with batch_norm being {use_batchnorm} and batch size of {batch_size}\n it ran for {num_of_epochs} epoches, {num_of_iterations} iterations and {num_of_training_steps} training steps before stoping.')
    print('Validation accuracy: {:.2f}%\nTrain Accuracy: {:.2f}%\nTest Accuracy: {:.2f}%'.format(val_acc, train_acc,
                                                                                                 test_acc))
    print('-' * 64)


(x_train, y_train), (x_test, y_test) = load_data()

x_test = x_test.reshape((x_test.shape[0], 784))
layers_dims = [784, 20, 7, 5, 10]
run_experiment(x_train, y_train, layers_dims,learning_rate=0.1, use_batchnorm=True)
run_experiment(x_train, y_train, layers_dims,learning_rate=0.1, use_batchnorm=False)
