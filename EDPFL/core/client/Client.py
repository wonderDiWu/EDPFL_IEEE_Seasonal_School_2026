# Client ABC
from abc import ABC, abstractmethod
class Client(ABC):
	@abstractmethod
	def initialize(self, *args, **kwargs):
		pass
	@abstractmethod
	def train(self, *args, **kwargs):
		pass
	@abstractmethod
	def upload(self, *args, **kwargs):
		pass