import uuid

from main.neuron.activation.activation_functors import AbstractActivationFunctor
from main.neuron.summators.abstract_summator import AbstractSummator
from main.neuron.summators.array.arrays_summator import ArraysSummator
from main.utils import check_class, Disposable


class NeuronInputProvider:

    def __init__(self):
        self._dispose_callback = None

    def get_input(self):
        raise NotImplementedError()

    def set_on_dispose(self, on_dispose):
        self._dispose_callback = on_dispose


class ConstantNeuronInputProvider(NeuronInputProvider):

    def __init__(self, value=0):
        super().__init__()
        self._value = value

    def get_input(self):
        return self._value

    def set_input_value(self, value):
        self._value = value


class Neuron(NeuronInputProvider, Disposable):

    def __init__(self, activator, summator=ArraysSummator(), input_providers=None):
        super().__init__()
        if input_providers is None:
            input_providers = []
        self._id = str(uuid.uuid4().hex)

        check_class(
            activator, AbstractActivationFunctor, "Activator parameter is not subtype of AbstractActivationFunctor"
        )
        self._activator = activator

        check_class(summator, AbstractSummator, "Summator parameter is not subtype of AbstractSummator")
        self._summator = summator

        self._input_providers = input_providers
        for provider in self._input_providers:
            provider.set_on_dispose(lambda: self._input_providers.remove(provider))

        self._last_result = 0.0

    def get_activator(self):
        return self._activator

    def add_input(self, input_provider):
        check_class(input_provider, WeightedInputProvider, "Input neuron should be a WeightedNeuron")
        input_provider.set_on_dispose(lambda: self._input_providers.remove(input_provider))
        self._input_providers.append(input_provider)

    def activate(self):
        weights = []
        inputs = []
        for weighted_provider in self._input_providers:
            weights.append(weighted_provider.get_weight())
            inputs.append(weighted_provider.get_input())

        self._last_result = self._activator.activate(self._summator.sum(weights, inputs))

    def get_id(self):
        return self._id

    def get_input(self):
        return self._last_result

    def dispose(self):
        self._dispose_callback()


class WeightedInputProvider:

    def __init__(self, weight, input_provider):
        self.__check_weight(weight)
        self._weight = weight

        check_class(
            input_provider, NeuronInputProvider, "Neuron parameter in WeightedNeuron isn't subtype of Neuron class"
        )
        self._input_provider = input_provider

    @staticmethod
    def __check_weight(weight):
        check_class(weight, (int, float), "Weight of neuron should be double or int")

    def get_weight(self):
        return self._weight

    def set_weight(self, weight):
        self.__check_weight(weight)
        self._weight = weight

    def set_on_dispose(self, on_dispose):
        self._input_provider.set_on_dispose(on_dispose)

    def get_input(self):
        self._input_provider.get_input()
