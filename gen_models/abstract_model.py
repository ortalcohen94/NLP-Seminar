import abc
import torch

class GenModel (abc.ABC):
    @abc.abstractmethod
    def sample_instance (self, original_sentence: torch.Tensor):
        raise NotImplementedError()
    
    # @abc.abstractmethod
    # def train(self, dl_train, dl_test, num_epochs):
    #     raise NotImplemented()