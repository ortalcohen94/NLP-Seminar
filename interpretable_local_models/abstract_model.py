import abc


class LocalModel (abc.ABC):
    @abc.abstractmethod
    def train(self, batch):
        raise NotImplementedError()

    def get_explanation(self):
        raise NotImplementedError()
    