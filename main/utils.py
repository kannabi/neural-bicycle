class ClassTypeException(Exception):
    pass


def check_class(val, expected_type, exception_text=""):
    if not isinstance(val, expected_type):
        raise ClassTypeException(exception_text)


def on_dispose():
    pass


class Disposable:

    def dispose(self):
        """
        При вызове этого метода нейрон вызывает коллбек-функцию,
        чем извещает своих "подписчиков" о том, что от него надо отписаться
        """
        pass
