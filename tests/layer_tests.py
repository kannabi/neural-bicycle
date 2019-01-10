import unittest

from main.network.layer import Layer
from main.neuron.activation.activation_functors import ConstantActivationFunctor
from main.neuron.neuron import Neuron, ConstantNeuronInputProvider, WeightedInputProvider


class TestLayer(unittest.TestCase):

    def test_neuron_creating(self):
        layer = Layer()
        created_neuron_id = layer.create_neuron(ConstantActivationFunctor())
        self.assertEqual(len(layer.get_neurons()), 1, "There is wrong number of all neurons")
        self.assertEqual(layer.get_dimension(), 1, "There is wrong dimension. It should be 1")
        self.assertNotEqual(layer.get_neuron(created_neuron_id), None, "Can't get created neuron")

    def test_input_adding_and_activation(self):
        layer = Layer()
        layer_neuron_id = layer.create_neuron(ConstantActivationFunctor())

        layer.add_input_to_neuron(layer_neuron_id, WeightedInputProvider(1, ConstantNeuronInputProvider(5)))

        output_neuron = Neuron(ConstantActivationFunctor())
        output_neuron.add_input(
            WeightedInputProvider(1, layer.get_neuron(layer_neuron_id))
        )

        layer.activate_layer()
        output_neuron.activate()
        self.assertEqual(output_neuron.get_input(), 5, "Layer activation result isn't correct")

    def test_input_remove(self):
        layer = Layer()
        layer_neuron_id = layer.create_neuron(ConstantActivationFunctor())

        provider = ConstantNeuronInputProvider(5)
        provider_id = provider.get_id()
        layer.add_input_to_neuron(layer_neuron_id, WeightedInputProvider(1, provider))
        layer.remove_input_from_neuron(layer_neuron_id, provider_id)

        output_neuron = Neuron(ConstantActivationFunctor())
        output_neuron.add_input(
            WeightedInputProvider(1, layer.get_neuron(layer_neuron_id))
        )

        layer.activate_layer()
        output_neuron.activate()
        self.assertEqual(output_neuron.get_input(), 0, "Layer activation result isn't correct")

