from exceptions import SummatorClassCastException, ActivatorClassCastException
from summators.abstract_summator import AbstractSummator


class AbstractActivationFunctor:

    def activate(self, activation_input):
        raise NotImplementedError("Ooops! There is not implemented activation functor!")


class Neuron:

    def __init__(self, activator, summator):
        if not isinstance(activator, AbstractActivationFunctor):
            raise ActivatorClassCastException()
        self._activator = activator

        if not isinstance(summator, AbstractSummator):
            raise SummatorClassCastException()
        self._summator = summator

    def activate(self, w, x):
        return self._activator.activate(self._summator.sum(w, x))
