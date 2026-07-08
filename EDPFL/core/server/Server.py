# Server ABC
from abc import ABC, abstractmethod

class Server(ABC):
	@abstractmethod
	def initialize(self, *args, **kwargs):
		pass
	@abstractmethod
	def train(self, *args, **kwargs):
		pass
	@abstractmethod
	def aggregate(self, *args, **kwargs):
		pass
	@abstractmethod
	def test(self, *args, **kwargs):
		pass