import unittest

from main.neuron.activation.activation_functors import ConstantActivationFunctor
from main.neuron.neuron import Neuron, ConstantNeuronInputProvider, WeightedInputProvider


class TestNeuron(unittest.TestCase):

    def test_adding_input(self):
        neuron = Neuron(ConstantActivationFunctor())
        neuron.add_input(WeightedInputProvider(1, ConstantNeuronInputProvider(5)))
        neuron.activate()
        self.assertEqual(neuron.get_input(), 5, "Neuron activation result isn't correct")

    def test_adding_two_input(self):
        neuron = Neuron(ConstantActivationFunctor())
        neuron.add_input(WeightedInputProvider(1, ConstantNeuronInputProvider(5)))
        neuron.add_input(WeightedInputProvider(1, ConstantNeuronInputProvider(5)))
        neuron.activate()
        self.assertEqual(neuron.get_input(), 10, "Neuron activation result isn't correct")

    def test_input_disposing(self):
        input_neuron = Neuron(ConstantActivationFunctor())
        target_neuron0 = Neuron(ConstantActivationFunctor())
        target_neuron1 = Neuron(ConstantActivationFunctor())

        target_neuron0.add_input(WeightedInputProvider(1, input_neuron))
        target_neuron1.add_input(WeightedInputProvider(1, input_neuron))

        input_neuron.dispose()

        target_neuron0.activate()
        target_neuron1.activate()
        self.assertEqual(target_neuron0.get_input(), 0, "Neuron activation result isn't correct")
        self.assertEqual(target_neuron1.get_input(), 0, "Neuron activation result isn't correct")

