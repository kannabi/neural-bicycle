import unittest

from main.network.network import Network
from main.neuron.activation.activation_functors import ConstantActivationFunctor


class TestNetwork(unittest.TestCase):

    def test_simple_activation(self):
        network = Network()
        input_neuron_id = network.add_input_neuron()

        network.add_layer()
        hidden_neuron_id = network.add_neuron(0, ConstantActivationFunctor())
        network.link_neurons(input_neuron_id, 0, hidden_neuron_id)

        output_neuron_id = network.add_output_neuron(ConstantActivationFunctor())
        network.add_output_input(hidden_neuron_id, output_neuron_id)

        res = network.activate([1])
        self.assertEqual(res[0], 1, "")
