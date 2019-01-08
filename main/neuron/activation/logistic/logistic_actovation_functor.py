from math import exp

from main.neuron.activation.abstract_activation_functor import AbstractActivationFunctor


class LogisticActivationFunctor(AbstractActivationFunctor):

    def activate(self, activation_input):
        return 1 / (1 + exp(activation_input))
