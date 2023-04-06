import abc

class Loss(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, data, **kwargs):
        pass