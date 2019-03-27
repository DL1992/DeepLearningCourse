import numpy as np
import math


def initialize_parameters(layer_dims):
    """

    :param layer_dims: an array of the dimensions of each layer in the network (layer 0 is the size of the
            flattened input, layer L is the output sigmoid)
    :return: parameters - a dictionary containing the initialized W and b parameters of each layer
        (W1…WL, b1…bL).
    """
    parameters = {('W' + str(l)): np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01 for l in
                  range(1, len(layer_dims))}
    bias = {('b' + str(l)): np.zeros((layer_dims[l], 1)) for l in range(1, len(layer_dims))}
    parameters.update(bias)
    return parameters


def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    :param A: the activations of the previous layer
    :param W: the weight matrix of the current layer (of shape [size of current layer, size of previous layer])
    :param b: the bias vector of the current layer (of shape [size of current layer, 1])
    :return: a tuple(Z, linear_cache) -
                Z – the linear component of the activation function (i.e., the value before applying the
                non-linear function)
                linear_cache – a dictionary containing A, W, b (stored for making the backpropagation
    """
    Z = W @ A + b
    linear_cache = (A, W, b)
    return Z, linear_cache

def softmax(X):
    """

    :param X: prediction values
    :return: normalize prediction values
    """
    z = X.T
    exps = np.exp(z - np.max(z))
    A = exps / np.sum(exps)
    A = A.T

    return A


def sigmoid(Z):
    """
    :param Z: the linear component of the activation function
    :return: a tuple(A, activation_cache) -
            A – the activations of the layer
            activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = 1 / (1 + np.exp(-Z))
    activation_cache = Z
    return A, activation_cache


def relu(Z):
    """
    :param Z: the linear component of the activation function
    :return: a tuple(A, activation_cache) -
            A – the activations of the layer
            activation_cache – returns Z, which will be useful for the backpropagation
    """
    A = np.maximum(0, Z)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation='relu'):
    """

    :param A_prev: activations of the previous layer
    :param W: the weights matrix of the current layer
    :param B: the bias vector of the current layer
    :param activation: the activation function to be used (a string, either “sigmoid” or “relu”)
    :return: a tuple(A,cache)-
            A – the activations of the current layer
            cache – a joint dictionary containing both linear_cache and activation_cache
    """
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == 'sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation == 'relu':
        A, activation_cache = relu(Z)
    elif activation == 'softamx':
        A, activation_cache = softmax(Z)
    else:
        raise ValueError('No such activation')

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters, use_batchnorm=False):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID
    computation.

    :param X: the data, numpy array of shape (input size, number of examples)
    :param parameters: the initialized W and b parameters of each layer
    :param use_batchnorm: a boolean flag used to determine whether to apply batchnorm after
            the activation
    :return: a tuple(AL, caches) -
                AL – the last post-activation value
                caches – a list of all the cache objects generated by the linear_forward function
    """
    caches = []
    A = X
    n_layers = len(parameters) // 2
    for l in range(1, n_layers):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters['W' + str(l)],
                                             parameters['b' + str(l)],
                                             activation='relu')
        if use_batchnorm:
            apply_batchnorm(A)
        caches.append(cache)
    # AL, cache = linear_activation_forward(A,
    #                                       parameters['W' + str(n_layers)],
    #                                       parameters['b' + str(n_layers)],
    #                                       activation='softmax')
    AL, cache = linear_activation_forward(A,
                                          parameters['W' + str(n_layers)],
                                          parameters['b' + str(n_layers)],
                                          activation='sigmoid')

    caches.append(cache)

    return AL, caches


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation.
    𝑐𝑜𝑠𝑡 = − '( ∗ Σ [,𝑦. ∗ log(𝐴𝐿)6 + (,1 − 𝑦.6 ∗ log(1 − 𝐴𝐿))]

    :param AL: probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: cost – the cross-entropy cost
    """

    m = Y.shape[1]
    cost = np.sum(Y * np.log(AL))
    cost /= -m
    return cost

def compute_cost_new(AL, Y):
    """
    Implement the cost function defined by equation.
    𝑐𝑜𝑠𝑡 = − '( ∗ Σ [,𝑦. ∗ log(𝐴𝐿)6 + (,1 − 𝑦.6 ∗ log(1 − 𝐴𝐿))]

    :param AL: probability vector corresponding to your label predictions, shape (1, number of examples)
    :param Y: the labels vector (i.e. the ground truth)
    :return: cost – the cross-entropy cost
    """

    # m = Y.shape[1]
    # cost = np.sum(Y * np.log(AL))
    # cost /= -m
    # return cost
    pass

def apply_batchnorm(A):
    """
    performs batchnorm on the received activation values of a given layer.

    :param A: the activation values of a given layer
    :return: NA - the normalized activation values, based on the formula learned in class
    """
    gamma = 1
    beta = 0
    mu = np.mean(A)
    var = np.var(A)

    A_norm = (A - mu) / np.sqrt(var + np.exp(-8))
    NA = gamma * A_norm + beta

    cache = A, A_norm, mu, var, gamma, beta
    return NA, cache


def Linear_backward(dZ, cache):
    """
    Implements the linear part of the backward propagation process for a single layer.

    :param dZ: the gradient of the cost with respect to the linear output of the current layer (layer l)
    :param cache: tuple of values (A_prev, W, b) coming from the forward propagation in the current layer
    :return: a tuple(dA_prev, dW,db) -
            dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW -- Gradient of the cost with respect to W (current layer l), same shape as W
            db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, A_prev.T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation='relu'):
    """
    Implements the backward propagation for the LINEAR->ACTIVATION layer. The function
    first computes dZ and then applies the linear_backward function.

    :param dA: post activation gradient of the current layer
    :param cache: contains both the linear cache and the activations cache
    :param activation:
    :return: a tuple(dA_prev,dW,db)-
            dA_prev – Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
            dW – Gradient of the cost with respect to W (current layer l), same shape as W
            db – Gradient of the cost with respect to b (current layer l), same shape as b
    """
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == 'softmax':
        dZ = softmax_backward(dA, activation_cache)

    return Linear_backward(dZ, linear_cache)

def softmax_backward(dA, activation_cache):
    """
       Implements backward propagation for a softamx unit

       :param dA: the post-activation gradient
       :param activation_cache: contains Z (stored during the forward propagation)
       :return: dZ – gradient of the cost with respect to Z
       """
    pass

def relu_backward(dA, activation_cache):
    """
    Implements backward propagation for a ReLU unit

    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    return np.where(activation_cache <= 0, 0, dA)


def sigmoid_backward(dA, activation_cache):
    """
    Implements backward propagation for a sigmoid unit

    :param dA: the post-activation gradient
    :param activation_cache: contains Z (stored during the forward propagation)
    :return: dZ – gradient of the cost with respect to Z
    """
    sig_z = 1 / (1 + np.exp(-activation_cache))
    return dA * (sig_z * (1 - sig_z))


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation process for the entire network.

    :param AL: the probabilities vector, the output of the forward propagation (L_model_forward)
    :param Y: the true labels vector (the "ground truth" - true classifications)
    :param caches: ist of caches containing for each layer: a) the linear cache; b) the activation cache
    :return: Grads - a dictionary with the gradients
                    grads["dA" + str(l)] = ...
                    grads["dW" + str(l)] = ...
                    grads["db" + str(l)] = ...
    """

    Grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL + 1e-13) - np.divide(1 - Y, 1 - AL + 1e-13))

    cache = caches[-1]
    Grads['dA' + str(L)], Grads['dW' + str(L)], Grads['db' + str(L)] = linear_activation_backward(dAL,
                                                                                                  cache,
                                                                                                  activation='sigmoid')

    # Grads['dA' + str(L)], Grads['dW' + str(L)], Grads['db' + str(L)] = linear_activation_backward(dAL,
    #                                                                                               cache,
    #
    for l in reversed(range(L - 1)):
        cache = caches[l]
        dAL = Grads['dA' + str(l + 2)]
        Grads['dA' + str(l + 1)], Grads['dW' + str(l + 1)], Grads['db' + str(l + 1)] = linear_activation_backward(dAL,
                                                                                                                  cache,
                                                                                                                  activation='relu')
    return Grads


def Update_parameters(parameters, grads, learning_rate=0.001):
    """
    Updates parameters using gradient descent

    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :param grads: a python dictionary containing the gradients (generated by L_model_backward)
    :param learning_rate: the learning rate used to update the parameters (the “alpha”)
    :return: parameters – the updated values of the parameters object provided as input
    """

    new_parameters = {key: value - learning_rate * grads['d' + key] for (key, value) in parameters.items()}
    return new_parameters

def val_split(X, Y, split_prob=0.8):
    validation_split = np.random.rand(len(X)) < split_prob

    x_train_data = X[validation_split]
    x_val_data = X[~validation_split]
    y_train_data = Y[:, validation_split]
    y_val_data = Y[:, ~validation_split]

    x_train_data = x_train_data.reshape((x_train_data.shape[0], 784))
    x_val_data = x_val_data.reshape((x_val_data.shape[0], 784))
    return x_train_data.T, y_train_data, x_val_data.T, y_val_data

def L_layer_model(X, Y, layer_dims, learning_rate=0.009, num_iterations=300, batch_size=32, use_batchnorm=False):
    """
    Implements a L-layer neural network. All layers but the last should have the ReLU
    activation function, and the final layer will apply the sigmoid activation function. The
    size of the output layer should be equal to the number of labels in the data.

    :param X: the input data, a numpy array of shape (height*width , number_of_examples).
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples).
    :param layer_dims: a list containing the dimensions of each layer, including the input.
    :param learning_rate: the learning rate used to update the parameters (the “alpha”).
    :param num_iterations: the number of iteration to train the model for.
    :param batch_size: the number of examples in a single training batch.
    :return: a tuple(parameters,costs) -
            parameters – the parameters learnt by the system during the training (the same parameters that were updated
            in the update_parameters function).
            costs – the values of the cost function (calculated by the compute_cost function). One value is to be saved
             after each 100 training iterations (e.g. 3000 iterations -> 30 values).
    """
    num_of_epoches = 0
    num_of_training_steps = 0
    no_improve_count = 0
    curr_acc = 0.0
    early_stop = False
    X, Y, val_X, val_Y = val_split(X, Y)

    costs = []
    parameters = initialize_parameters(layer_dims)
    while not early_stop:
        num_of_epoches += 1
        for i in range(num_iterations):
            print("num of iter" + str(i))
            minibatches = mini_batches_split(X, Y, batch_size)
            for minibatch in minibatches:
                if early_stop:
                    break
                num_of_training_steps += 1
                x, y = minibatch
                AL, caches = L_model_forward(x, parameters, use_batchnorm)
                grads = L_model_backward(AL, y, caches)
                parameters = Update_parameters(parameters, grads, learning_rate)

                acc = Predict(val_X, val_Y, parameters) * 100
                if acc  < curr_acc + 1e-8:
                    no_improve_count += 1
                else:
                    curr_acc = acc
                    no_improve_count = 0
                if no_improve_count == 100000:
                    early_stop = True

                if num_of_training_steps % 100 == 0:
                    cost = compute_cost(AL, y)
                    # acc = Predict(val_X, val_Y, parameters) * 100
                    costs.append(cost)
                    print(f"Cost after num_of_training_steps {num_of_training_steps}: {cost}, {curr_acc}%")
            if early_stop:
                break

    return parameters, costs, num_of_epoches, num_of_training_steps, curr_acc


def Predict(X, Y, parameters):
    """
    The function receives an input data and the true labels and calculates the accuracy of
    the trained neural network on the data.

    :param X: the input data, a numpy array of shape (height*width, number_of_examples)
    :param Y: the “real” labels of the data, a vector of shape (num_of_classes, number of examples)
    :param parameters: a python dictionary containing the DNN architecture’s parameters
    :return: accuracy – the accuracy measure of the neural net on the provided data (i.e. the
                percentage of the samples for which the correct label receives over 50% of the
                confidence score). Use the softmax function to normalize the output values.
    """
    AL, _ = L_model_forward(X, parameters)
    soft_AL = softmax(AL)
    preds = np.argmax(soft_AL, axis=0)
    Y = np.argmax(Y, axis=0)
    acc = (preds - Y == 0).sum()
    acc /= X.shape[1]
    return acc


def mini_batches_split(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)
    :param X: input data, of shape (input size, number of examples)
    :param Y: true "label"
    :param mini_batch_size: size of the mini-batches
    :return: mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    # shuffled_Y = Y[permutation]
    shuffled_Y = Y[:, permutation].reshape((10, m))
    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        # mini_batch_Y = shuffled_Y[k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size:]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches
