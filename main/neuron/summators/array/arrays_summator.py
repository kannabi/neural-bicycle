from functools import reduce

from main.neuron.summators.abstract_summator import AbstractSummator
from main.neuron.summators.array.exceptions import WrongSummatorParametersException, WrongSummatorVectorLengthsException


class ArraysSummator(AbstractSummator):

    def sum(self, v0, v1):
        v0_is_digit = self.__is_digit(v0)
        v0_is_vector = self.__is_vector(v0)

        v1_is_digit = self.__is_digit(v1)
        v1_is_vector = self.__is_vector(v1)

        if v0_is_digit and v1_is_vector:
            return self.__multiply_vector_by_number(v0, v1)
        elif v0_is_vector and v1_is_digit:
            return self.__multiply_vector_by_number(v1, v0)
        elif v0_is_vector and v1_is_vector:
            return self.__multiply_vectors(v0, v1)
        elif v0_is_digit and v1_is_digit:
            return v0 * v1
        else:
            raise WrongSummatorParametersException()

    @staticmethod
    def __is_digit(val):
        return isinstance(val, (int, float, complex))

    def __is_vector(self, val):
        return (isinstance(val, list) or isinstance(val, tuple)) \
               and reduce(lambda reducer, i: reducer and self.__is_digit(i), val, True)

    @staticmethod
    def __multiply_vector_by_number(number, vector):
        return list(map(lambda i: i * number, vector))

    def __multiply_vectors(self, v0, v1):
        self.__check_vectors_length(v0, v1)
        return reduce(lambda s, t: s + t[0] * t[1], zip(v0, v1), 0)

    @staticmethod
    def __check_vectors_length(v0, v1):
        if len(v0) != len(v1):
            raise WrongSummatorVectorLengthsException()
