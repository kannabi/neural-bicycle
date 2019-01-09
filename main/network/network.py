from main.network.layer import Layer
from main.neuron.activation.activation_functors import ConstantActivationFunctor
from main.neuron.neuron import WeightedInputProvider, ConstantNeuronInputProvider, Neuron
from main.utils import check_class


class Network:

    def __init__(self, input_layer_dimension, output_layer_dimension, layers=None):
        check_class(input_layer_dimension, int)
        check_class(output_layer_dimension, int)
        if layers is None:
            layers = []
        self._layers = layers

        if len(layers) != 0:
            for layer in layers:
                check_class(layer, Layer)

        self._input_layer_providers = self.__create_initial_input_providers(input_layer_dimension)
        self._input_layer = self.__create_input_layer()
        self._output_layer = Layer()

    @staticmethod
    def __create_initial_input_providers(dimension):
        return list(map(lambda i: WeightedInputProvider(1, ConstantNeuronInputProvider()), range(dimension)))

    def __create_input_layer(self):
        return Layer(
            list(
                map(
                    lambda provider: Neuron(ConstantActivationFunctor(), input_providers=[provider]),
                    self._input_layer_providers
                )
            )
        )

    def activate(self, values):
        if len(values) != len(self._input_layer_providers):
            raise Exception("Input data size isn't fit with a input layer dimension")

        for i, value in enumerate(values):
            self._input_layer_providers[i].get_input().set_input_value(value)

        self._input_layer.activate_layer()

        for layer in self._layers:
            layer.activate_layer()

        self._output_layer.activate_layer()
