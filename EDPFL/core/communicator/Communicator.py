# Communicator ABC
from abc import ABC, abstractmethod
class Communicator(ABC):
    @abstractmethod
    def send_msg(self, *args, **kwargs):
        pass
    @abstractmethod
    def recv_msg(self, *args, **kwargs):
        pass