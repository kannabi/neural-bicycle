import uuid

from main.neuron.activation.abstract_activation_functor import AbstractActivationFunctor
from main.neuron.summators.abstract_summator import AbstractSummator
from main.utils import check_class


class Neuron:

    def __init__(self, activator, summator, **kwargs):
        self._id = str(uuid.uuid4().hex)

        check_class(
            activator, AbstractActivationFunctor, "Activator parameter is not subtype of AbstractActivationFunctor"
        )
        self._activator = activator

        check_class(summator, AbstractSummator, "Summator parameter is not subtype of AbstractSummator")
        self._summator = summator

        self._input_neurons = kwargs.get('input_neurons', [])

        self._last_result = 0.0

    def add_input_neuron(self, input_neuron):
        check_class(input_neuron, WeightedNeuron, "Input neuron should be a WeightedNeuron")
        self._input_neurons.append(input_neuron)

    def activate(self):
        weights = []
        inputs = []
        for neuron in self._input_neurons:
            weights.append(neuron.get_weight())
            inputs.append(neuron.get_neuron.get_last_result)

        self._last_result = self._activator.activate(self._summator.sum(weights, inputs))

    def get_id(self):
        return self._id

    def get_last_result(self):
        return self._last_result


class WeightedNeuron:

    def __init__(self, weight, neuron):

        self.__check_weight(weight)
        self._weight = weight

        check_class(neuron, Neuron, "Neuron parameter in WeightedNeuron isn't subtype of Neuron class")
        self._neuron = neuron

    @staticmethod
    def __check_weight(weight):
        check_class(weight, (int, float), "Weight of neuron should be double or int")

    def get_weight(self):
        return self._weight

    def set_weight(self, weight):
        self.__check_weight(weight)
        self._weight = weight

    def get_neuron(self):
        return self._neuron
