class ActivatorClassCastException(Exception):
    def __init__(self):
        super(ActivatorClassCastException, self).__init__(
            "Activator parameter is not subtype of AbstractActivationFunctor"
        )


class SummatorClassCastException(Exception):
    def __init__(self):
        super(SummatorClassCastException, self).__init__(
            "Summator parameter is not subtype of AbstractSummator"
        )
