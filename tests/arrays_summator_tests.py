import unittest

from main.neuron.summators.array.arrays_summator import ArraysSummator
from main.neuron.summators.array.exceptions import WrongSummatorVectorLengthsException, WrongSummatorParametersException


class TestArraysSummator(unittest.TestCase):

    def setUp(self):
        self._summator = ArraysSummator()

    def test_number_and_vector(self):
        res = self._summator.sum(2, [4, 8])
        self.assertEqual(res, [8, 16], "Number and vector is't correct")

    def test_vector_and_number(self):
        res = self._summator.sum([4, 8], 2)
        self.assertEqual(res, [8, 16], "Vector and number is't correct")

    def test_vector_and_vector(self):
        res = self._summator.sum([4, 8], [1, 2])
        self.assertEqual(res, 20, "Vector and vector is't correct")

    def test_number_and_number(self):
        res = self._summator.sum(4, 4)
        self.assertEqual(res, 16, "Vector and vector is't correct")

    def test_different_vectors_lengths(self):
        self.assertRaises(WrongSummatorVectorLengthsException, self._summator.sum, [], [1])

    def test_first_arg_trash(self):
        self.assertRaises(WrongSummatorParametersException, self._summator.sum, "", 0)
        self.assertRaises(WrongSummatorParametersException, self._summator.sum, "", "")
        self.assertRaises(WrongSummatorParametersException, self._summator.sum, 0, "")
