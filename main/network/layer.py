from collections import OrderedDict

from main.neuron.neuron import Neuron
from main.neuron.summators.array.arrays_summator import ArraysSummator


class Layer:

    def __init__(self):
        self._neurons = OrderedDict()

    def create_neuron(self, activator, summator=ArraysSummator()):
        neuron = Neuron(activator, summator)
        self._neurons[neuron.get_id()] = neuron
        return neuron.get_id()

    def get_neuron(self, neuron_id):
        return self._neurons[neuron_id]

    def get_neurons_id(self):
        return self._neurons.keys()

    def get_neurons(self):
        return self._neurons.values()

    def add_input_to_neuron(self, neuron_id, input_provider):
        neuron = self._neurons[neuron_id]
        if neuron is not None:
            neuron.add_input(input_provider)

    def remove_input_from_neuron(self, neuron_id, input_provider_id):
        neuron = self._neurons[neuron_id]
        if neuron is not None:
            neuron.remove_input(input_provider_id)

    def get_dimension(self):
        return len(self._neurons)

    def activate_layer(self):
        for neuron in self._neurons.values():
            neuron.activate()



