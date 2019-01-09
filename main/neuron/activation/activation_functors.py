from math import exp


class AbstractActivationFunctor:

    def activate(self, activation_input=0):
        raise NotImplementedError("Ooops! There is not implemented activation functor!")


class LogisticActivationFunctor(AbstractActivationFunctor):

    def activate(self, activation_input=0):
        return 1 / (1 + exp(activation_input))


class ConstantActivationFunctor(AbstractActivationFunctor):

    def activate(self, activation_input=0):
        return activation_input
