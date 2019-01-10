from main.network.layer import Layer
from main.neuron.activation.activation_functors import ConstantActivationFunctor
from main.neuron.neuron import WeightedInputProvider, ConstantNeuronInputProvider, Neuron
from main.neuron.summators.array.arrays_summator import ArraysSummator
from main.utils import check_class


class Network:

    def __init__(self, input_layer_dimension, output_layer_dimension):
        check_class(input_layer_dimension, int)
        check_class(output_layer_dimension, int)
        self._layers = []

        self._input_layer_providers = self.__create_initial_input_providers(input_layer_dimension)
        self._input_layer = Layer()
        self._output_layer = Layer()

    @staticmethod
    def __create_initial_input_providers(dimension):
        return list(map(lambda i: WeightedInputProvider(1, ConstantNeuronInputProvider()), range(dimension)))

    def add_layer(self, index=None):
        if index is None:
            self.__add_layer_to_end()
        else:
            self._layers.insert(index, Layer())

    def __add_layer_to_end(self):
        layers_len = len(self._layers)
        if layers_len != 0:
            self.__drop_all_layer_connection(self._layers[layers_len - 1])

        self._layers.append(Layer())

    def __add_layer_to_middle(self, index):
        self.__drop_all_layer_connection(self.__get_input_layer(index - 1))
        self._layers.insert(index, Layer)

    def delete_layer(self, layer_index):
        layer = self._layers[layer_index]
        for neuron in layer.get_neurons():
            neuron.dispose()
        self._layers.pop(layer_index)

    @staticmethod
    def __drop_all_layer_connection(layer):
        last_layer_neurons = layer.get_neurons()
        for neuron in last_layer_neurons:
            neuron.dispose()

    def get_hidden_layer_number(self):
        return len(self._layers)

    def add_neuron(self, layer_index, activator, summator=ArraysSummator()):
        return self._layers[layer_index].create_neuron(self, activator, summator)

    def link_neurons(self, input_neuron_id, target_layer_index, target_neuron_id):
        input_layer = self.__get_input_layer(target_layer_index)
        neuron = input_layer.get_neuron(input_neuron_id)
        if neuron is None:
            raise Exception("There is no such left neuron")
        self._layers[target_layer_index].add_input_to_neuron(
            target_neuron_id,
            WeightedInputProvider(1, neuron)
        )

    def unlink_neurons(self, input_neuron_id, target_layer_index, target_neuron_id):
        input_layer = self.__get_input_layer(target_layer_index)
        neuron = input_layer.get_neuron(input_neuron_id)
        if neuron is None:
            raise Exception("There is no such left neuron")
        self._layers[target_layer_index].remove_input_from_neuron(target_neuron_id, neuron.get_id())

    def __get_input_layer(self, index):
        return self._input_layer if index == 0 else self._layers[index - 1]

    def unlink_neuron_from_all(self, input_neuron_id, layer_index):
        input_layer = self.__get_input_layer(layer_index)
        neuron = input_layer.get_neuron(input_neuron_id)
        if neuron is None:
            raise Exception("There is no such left neuron")
        neuron.dispose()

    def create_full_connection(self, target_layer_index):
        input_layer = self._input_layer if target_layer_index == 0 else self._layers[target_layer_index - 1]
        input_neurons = input_layer.get_neurons()
        target_layer = self._layers[target_layer_index]
        for target_neuron_id in target_layer.get_neurons_id():
            for input_neuron in input_neurons:
                target_layer.add_input_to_neuron(
                    target_neuron_id,
                    WeightedInputProvider(1, input_neuron)
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
