from main.neuron.activation.abstract_activation_functor import AbstractActivationFunctor
from main.neuron.exceptions import SummatorClassCastException, ActivatorClassCastException
from main.neuron.summators.abstract_summator import AbstractSummator


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
