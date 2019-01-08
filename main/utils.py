class ClassTypeException(Exception):
    pass


def check_class(val, expected_type, exception_text=""):
    if not isinstance(val, expected_type):
        raise ClassTypeException(exception_text)
