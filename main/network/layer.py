from main.neuron.neuron import Neuron
from main.utils import check_class


class Layer:

    def __init__(self, neurons=None):
        if neurons is None:
            neurons = []
        self._neurons = {neuron.get_id(): neuron for neuron in neurons}

    def add_neuron(self, neuron):
        check_class(neuron, Neuron)
        self._neurons.update({neuron.get_id(), neuron})

    def get_neuron(self, neuron_id):
        check_class(neuron_id, str)
        return self._neurons.get(neuron_id)

    def get_neurons(self):
        return list(self._neurons.values())

    def get_dimension(self):
        return len(self._neurons)

    def activate_layer(self):
        for neuron in self._neurons.values():
            neuron.activate()



