import numpy as np
from utils import get_batches, plot_graph
from filereader import get_data


class Model:
    def __init__(self, parameters):
        '''
        The constructor function to setup initial values and required functions.
        :param parameters: A dictionary containing
                           1. 'num_layers': The number of layers in the network,
                           2. 'activation': A python list of length=num_layers of strings ('elu' and 'softmax') indicating
                                         activation of each layer,
                           3. 'loss': The loss function ('softmax' for softmax loss or categorical cross entropy loss function),
                           4. 'num_classes': The number of classes or categories in the label,
                           5. the weights and biases. Weight key = W<layer_number(1 indexed)>, bias key = b<layer_number(1 indexed)>,
        '''
        self.parameters = parameters
        self.activation = {
            'elu': self.elu,
            'softmax': self.softmax
        }
        self.d_activation = {
            'elu': self.d_elu,
            'softmax': self.d_softmax
        }
        self.loss = {
            'softmax': self.cross_entropy_loss
        }
        self.d_loss = {
            'softmax': self.d_cross_entropy_loss
        }
        self.batch_size = 256

    @staticmethod
    def softmax(Z):
        '''
        A function to compute the softmax activation
        :param Z:[numpy array]: Array of floats
        :return:[numpy array]: Array of floats, after application of softmax function to Z
        '''
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    @staticmethod
    def d_softmax(Z):
        '''
        A function to compute the derivative values of softmax activation
        :param Z:[numpy array]: Array of floats
        :return:[numpy array]: Array of floats, values corresponding to the derivative of softmax activation on Z
        '''
        return Z * (1 - Z)

    @staticmethod
    def elu(Z, alpha=1.2):
        '''
        A function to compute the elu(exponential linear unit) activation values.
        :param Z:[numpy array]: Array of floats, the score values.
        :param alpha:[float default=1.2]: the value for elu alpha
        :return:[numpy array]: elu activated values
        '''
        return np.where(Z >= 0, Z, alpha*(np.exp(Z) - 1))

    @staticmethod
    def d_elu(Z, alpha=1.2):
        '''
        A function to compute the derivative of elu(exponential linear unit) activation values.
        :param Z:[numpy array]: Array of floats, the score values.
        :param alpha:[float default=1.2]: the value for elu alpha
        :return:[numpy array]: the required derivative values
        '''
        return (Z >= 0).astype(np.float32) + (Z < 0).astype(np.float32) * (Model.elu(Z) + alpha)

    @staticmethod
    def cross_entropy_loss(labels, predictions, epsilon=1e-8):
        '''
        The function to compute the categorical cross entropy loss, given training labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :param epsilon:[float default=1e-8]: A small value for applying clipping for stability
        :return:[float]: The computed value of loss.
        '''
        predictions /= np.sum(predictions, axis=0, keepdims=True)
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.sum(labels * np.log(predictions))

    @staticmethod
    def d_cross_entropy_loss(labels, predictions):
        '''
        The function to compute the derivative values of categorical cross entropy values, given labels and prediction
        :param labels:[numpy array]: Training labels
        :param predictions:[numpy array]: Predicted labels
        :return:[numpy array]: The computed derivatives of categorical cross entropy function.
        '''
        return labels - predictions

    def train(self, data, labels, epochs=50, learning_rate=0.001, batch_size=256, l2_penalty=1e-4,
              optimization="adam", amsgrad=True, epsilon=1e-8, beta1=0.9, beta2=0.999, correct_bias=False):
        '''
        The function to train the neural network given the data and labels.
        :param data:[numpy array]: The training data.
        :param labels:[numpy array]: The training labels.
        :param epochs:[int default=50]: number of epochs.
        :param learning_rate:[float default=0.001]: The learning rate eta.
        :param batch_size:[int default=256]: The size of each batch.
        :param l2_penalty:[float default=1e-4]: The l2 regularization penalty.
        :param optimization:[string or None default="adam"]: Indicating the optimization function to be used(adam or rmsprop or None).
        :param amsgrad:[boolean default=True]: Set to true to apply amsgrad update to adam.
        :param epsilon:[float default=1e-8]: adam epsilon value.
        :param beta1:[float default=0.9]: momentum beta value.
        :param beta2:[float default=0.999]: adam beta value for squared exponential weighted average.
        :param correct_bias:[boolean default=False]: Set to true to apply adam bias correction.
        :return: tuple(costs_per_iter(list of loss per iteration on batch data), cost_per_epoch(list of loss per epoch on whole data)).
        '''
        self.batch_size = batch_size
        if optimization is not None and optimization != "adam":
            amsgrad = False if amsgrad else amsgrad
            correct_bias = False if not amsgrad else correct_bias
        iter = 1
        costs_per_iter, costs_per_epoch = [], []
        momentum_cache, rmsprop_cache = {}, {}
        for epoch in range(epochs):
            for i, (x_batch, y_batch) in enumerate(get_batches(data, labels)):
                prediction, cache = self.forward_propagate(x_batch, save_cache=True)
                costs_per_iter.append(self.cross_entropy_loss(y_batch, prediction))

                grads = self.back_propagate(x_batch, y_batch, prediction, cache)
                if optimization == "adam":
                    momentum_cache = self.momentum(grads, momentum_cache=momentum_cache, beta=beta1)
                if optimization == "rmsprop" or optimization == "adam":
                    rmsprop_cache = self.rmsprop(grads, rmsprop_cache=rmsprop_cache, beta=beta2, amsgrad=amsgrad)
                self.apply_grads(grads, batch_size=batch_size, optimization=optimization, momentum_cache=momentum_cache,
                                 rmsprop_cache=rmsprop_cache, learning_rate=learning_rate, l2_penalty=l2_penalty,
                                 epsilon=epsilon, correct_bias=correct_bias, iter=iter)
            iter+=batch_size
            costs_per_epoch.append(self.cross_entropy_loss(labels, self.predict(data)))
            print("Epoch {}: Training accuracy = {}".format(epoch+1, self.evaluate(labels, self.predict(data))))
        return costs_per_iter, costs_per_epoch

    def predict(self, data):
        '''
        Function to make a prediction based on the trained model, given the data
        :param data:[numpy array]: test data, for cifar10 the shape is [3072, -1]
        :return:[numpy array]: predictions of shape [10, -1] for cifar10
        '''
        predictions = np.zeros(shape=(self.parameters['num_classes'], data.shape[1]))
        num_batches = data.shape[1]//self.batch_size
        for batch_num, x_batch in enumerate(get_batches(data, shuffle=False)):
            if batch_num <= num_batches - 1:
                predictions[:, batch_num*self.batch_size:(batch_num+1)*self.batch_size] = \
                    self.forward_propagate(x_batch, save_cache=False)
            else:
                predictions[:, batch_num*self.batch_size:] = self.forward_propagate(x_batch, save_cache=False)
        return predictions

    def forward_propagate(self, X, save_cache=False):
        '''
        A function to output of the network, given the input.
        :param X:[numpy array]: Training data
        :param save_cache:[boolean default=False]: Set to true while training, for a cache to be used for back propagation.
        :return:[tuple(numpy array, dict) or just numpy array]: Output of the network and optionally, the cache of
                intermediate values that can be used for computing back propagation terms
        '''
        cache = {}
        for layer in range(self.parameters['num_layers']):
            if layer==0:
                Z = np.dot(self.parameters['W' + str(layer + 1)], X) + self.parameters['b' + str(layer + 1)]
            else:
                Z = np.dot(self.parameters['W' + str(layer + 1)], A) + self.parameters['b' + str(layer + 1)]
            if save_cache:
                cache['Z' + str(layer+1)] = Z.copy()
                if layer != 0:
                    cache['A' + str(layer+1)] = A.copy()
            A = self.activation[self.parameters['activation'][layer]](Z)
        if save_cache:
            return A, cache
        else:
            return A

    def evaluate(self, labels, predictions):
        '''
        A function to compute the accuracy of the predictions on a scale of 0-1.
        :param labels:[numpy array]: Training labels (or testing/validation if available)
        :param predictions:[numpy array]: Predicted labels
        :return:[float]: a number between [0, 1] denoting the accuracy of the prediction
        '''
        return np.mean(np.argmax(labels, axis=0) == np.argmax(predictions, axis=0))

    def back_propagate(self, data, labels, prediction, cache):
        '''
        A function to compute the back propagation gradients
        :param data:[numpy array]: The training data
        :param labels:[numpy array]: The training labels
        :param prediction:[numpy array]: The prediction based on training data and the parameters
        :param cache:[dict]: The cache containing all the parameters
        :return:[dict]: cache containing the back propagated gradients.
        '''
        grads = {}
        batch_size = data.shape[1]
        d_A = self.d_loss[self.parameters['loss']](labels, prediction)
        for layer in range(self.parameters['num_layers']-1, -1, -1):
            d_Z = d_A * self.d_activation[self.parameters['activation'][layer]](cache['Z' + str(layer + 1)])
            if layer == 0:
                grads['dW' + str(layer + 1)] = np.dot(d_Z, data.T)/batch_size
            else:
                grads['dW' + str(layer + 1)] = np.dot(d_Z, cache['A' + str(layer + 1)].T) / batch_size
            grads['db' + str(layer + 1)] = np.sum(d_Z, axis=1, keepdims=True)
            d_A = np.dot(self.parameters['W' + str(layer + 1)].T, d_Z)
        return grads

    def init_cache(self):
        '''
        A function to initialize the gradient descent or optimization caches consistently.
        :return:[dict]: cache with zero initialized np arrays for each layer.
        '''
        cache = {}
        for layer in range(self.parameters['num_layers']):
            cache['dW' + str(layer + 1)] = np.zeros_like(self.parameters['W' + str(layer + 1)])
            cache['db' + str(layer + 1)] = np.zeros_like(self.parameters['b' + str(layer + 1)])
        return cache

    def momentum(self, grads, momentum_cache={}, beta=0.9):
        '''
        A function to compute the momentum exponentially weighted average [Ning Quian https://doi.org/10.1016/S0893-6080(98)00116-6]
        :param grads:[dict]: cache of gradients calculated based on back propagation.
        :param momentum_cache:[dict]: cache of momentum exponentially weighted average terms.
        :param beta:[float default=0.9]: momentum beta value.
        :return:[dict]: cache of updated momentum exponentially weighted average terms.
        '''
        if not momentum_cache:
            momentum_cache = self.init_cache()

        for layer in range(self.parameters['num_layers']):
            momentum_cache['dW' + str(layer + 1)] = beta*momentum_cache['dW' + str(layer + 1)] + \
                                                    (1-beta)*grads['dW' + str(layer + 1)]
            momentum_cache['db' + str(layer + 1)] = beta * momentum_cache['db' + str(layer + 1)] + \
                                                    (1 - beta) * grads['db' + str(layer + 1)]
        return momentum_cache

    def rmsprop(self, grads, rmsprop_cache={}, beta=0.999, amsgrad=False):
        '''
        A function to compute the rmsprop exponentially weighted average [unpublished Hinton et al. (revealed in Coursera)]
        :param grads:[dict]: cache of gradients calculated based on back propagation.
        :param rmsprop_cache:[dict]: cache of rmsprop exponentially weighted average terms.
        :param beta:[float default=0.999]: rmsprop beta value.
        :param amsgrad:[boolean default=False]: set True to apply amsgrad.
        :return:[dict]: cache of updated rmsprop exponentially weighted average terms
        '''
        if not rmsprop_cache:
            rmsprop_cache = self.init_cache()

        for layer in range(self.parameters['num_layers']):
            new_weights =  beta*rmsprop_cache['dW' + str(layer + 1)] + \
                                                   (1 - beta)*(grads['dW' + str(layer + 1)]**2)
            new_bias = beta * rmsprop_cache['db' + str(layer + 1)] + \
                                                  (1 - beta) * (grads['db' + str(layer + 1)] ** 2)
            if amsgrad:
                rmsprop_cache['dW' + str(layer + 1)] = np.maximum(new_weights, rmsprop_cache['dW' + str(layer + 1)])
                rmsprop_cache['db' + str(layer + 1)] = np.maximum(new_bias, rmsprop_cache['db' + str(layer + 1)])
            else:
                rmsprop_cache['dW' + str(layer + 1)] = new_weights
                rmsprop_cache['db' + str(layer + 1)] = new_bias
        return rmsprop_cache

    def apply_grads(self, grads, batch_size=256, optimization=None, momentum_cache=None, rmsprop_cache=None,
                    learning_rate=0.001, l2_penalty=1e-4, epsilon=1e-8,
                    correct_bias=False, beta1=0.9, beta2=0.999, iter=999):
        '''
        A function to update weights based on calculated gradients and optimization caches.
        :param grads:[dict]: cache containing back propagation gradients.
        :param batch_size:[int default=256]: The size of each batch.
        :param optimization:[string or None default="adam"]: Indicating the optimization function to be used(adam or rmsprop or None).
        :param momentum_cache:[dict]: cache of momentum exponentially weighted average terms.
        :param rmsprop_cache:[dict]: cache of rmsprop exponentially weighted average terms.
        :param learning_rate:learning_rate:[float default=0.001]: The learning rate eta.
        :param l2_penalty:[float default=1e-4]: The l2 regularization penalty.
        :param epsilon:[float default=1e-8]: adam epsilon value.
        :param correct_bias:[boolean default=False]: Set to true to apply adam bias correction.
        :param beta1:[float default=0.9]: momentum beta value.
        :param beta2:[float default=0.999]: adam beta value for squared exponential weighted average.
        :param iter:[int default=999]: indicates the iteration number.
        :return: Void
        '''

        for layer in range(self.parameters['num_layers']):
            if optimization is None:
                self.parameters["W" + str(layer + 1)] -= learning_rate*(grads['dW' + str(layer + 1)] +
                                                                        l2_penalty*self.parameters["W" + str(layer + 1)])
                self.parameters["b" + str(layer + 1)] -= learning_rate * (grads['db' + str(layer + 1)] +
                                                                          l2_penalty * self.parameters["b" + str(layer + 1)])
            if optimization == "rmsprop":
                W_learning_rate = learning_rate/(np.sqrt(rmsprop_cache['dW' + str(layer + 1)]) + epsilon)
                b_learning_rate = learning_rate/(np.sqrt(rmsprop_cache['db' + str(layer + 1)]) + epsilon)
                self.parameters["W" + str(layer + 1)] -= W_learning_rate * (grads['dW' + str(layer + 1)] +
                                                                          l2_penalty * self.parameters["W" + str(layer + 1)])
                self.parameters["b" + str(layer + 1)] -= b_learning_rate * (grads['db' + str(layer + 1)] +
                                                                          l2_penalty * self.parameters["b" + str(layer + 1)])
            if optimization == "adam":
                if correct_bias:
                    W_first_moment = momentum_cache['dW' + str(layer + 1)]/(1 - beta1**iter)
                    B_first_moment = momentum_cache['db' + str(layer + 1)]/(1 - beta1**iter)
                    W_second_moment = rmsprop_cache['dW' + str(layer + 1)]/(1 - beta2**iter)
                    B_second_moment = rmsprop_cache['db' + str(layer + 1)]/(1 - beta2**iter)
                else:
                    W_first_moment = momentum_cache['dW' + str(layer + 1)]
                    B_first_moment = momentum_cache['db' + str(layer + 1)]
                    W_second_moment = rmsprop_cache['dW' + str(layer + 1)]
                    B_second_moment = rmsprop_cache['db' + str(layer + 1)]

                W_learning_rate = learning_rate / (np.sqrt(W_second_moment) + epsilon)
                b_learning_rate = learning_rate / (np.sqrt(B_second_moment) + epsilon)

                self.parameters["W" + str(layer + 1)] -= W_learning_rate * (W_first_moment +
                                                                            l2_penalty * self.parameters[
                                                                                "W" + str(layer + 1)])
                self.parameters["b" + str(layer + 1)] -= b_learning_rate * (B_first_moment +
                                                                            l2_penalty * self.parameters[
                                                                                "b" + str(layer + 1)])


def he_initialize(fan_in, fan_out):
    '''
    A function from smart initialization of parameters [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array]: A randomly initialized array of shape [fan_out, fan_in]
    '''
    return np.random.normal(0, 1, size=(fan_out, fan_in))*np.sqrt(2/fan_in), np.zeros((fan_out, 1))


if __name__ == '__main__':
    train_data, train_labels = get_data(num_samples=50000)
    test_data, test_labels = get_data(num_samples=10000, dataset="testing")

    train_data = train_data.reshape(-1, 32*32*3).T/255
    test_data = test_data.reshape(-1, 32*32*3).T/255

    print("Training data shape:", train_data.shape)
    print("Training labels shape:", train_labels.shape)
    print("Testing data shape:", test_data.shape)
    print("Testing labels shape:", test_labels.shape)

    dims, samples = train_data.shape

    parameters = dict({
        'num_layers': 3,
        'activation': ['elu', 'elu', 'softmax'],
        'loss': 'softmax',
        'num_classes': 10
    })

    parameters['W1'], parameters["b1"] = he_initialize(dims, 200)
    parameters['W2'], parameters["b2"] = he_initialize(200, 200)
    parameters['W3'], parameters["b3"] = he_initialize(200, 10)

    model = Model(parameters=parameters)

    costs_per_iter, costs_per_epoch = model.train(train_data, train_labels, optimization="adam", learning_rate=0.001, correct_bias=True)
    plot_graph(costs_per_iter, xlabel="Iterations", ylabel="Cost", title="Variation of cost per iteration")
    plot_graph(costs_per_epoch, xlabel="Epochs", ylabel="Cost", title="Variation of cost per epoch")

    print("Testing accuracy = {}".format(model.evaluate(test_labels, model.predict(test_data))))
