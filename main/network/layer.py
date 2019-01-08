from main.neuron.neuron import Neuron
from main.utils import check_class


class Layer:

    def __init__(self, **kwargs):
        self._neurons = {neuron.get_id(): neuron for neuron in kwargs.get("neurons", [])}

    def add_neuron(self, neuron):
        check_class(neuron, Neuron)
        self._neurons.update({neuron.get_id(), neuron})

    def get_neuron(self, neuron_id):
        check_class(neuron_id, str)
        return self._neurons.get(neuron_id)

    def get_neurons(self):
        return list(self._neurons.values())

    def activate_layer(self):
        for neuron in self._neurons.values():
            neuron.activate()



