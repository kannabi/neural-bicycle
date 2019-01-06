class WrongSummatorParametersException(Exception):
    def __init__(self):
        super(WrongSummatorParametersException, self).__init__(
            "Summator's parameters is not vector or number"
        )


class WrongSummatorVectorLengthsException(Exception):
    def __init__(self):
        super(WrongSummatorVectorLengthsException, self).__init__(
            "Summator's vector parameters have different lengths"
        )
