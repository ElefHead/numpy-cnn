# Not currently in use.

import numpy as np


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, correct_bias=True, amsprop=True):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.correct_bias = correct_bias
        self.amsprop = amsprop
        self.rmsprop_cache = {}
        self.momentum_cache = {}

    def rmsprop(self):
        pass

    def momentum(self):
        pass

    def minimize(self, model, loss):
        pass