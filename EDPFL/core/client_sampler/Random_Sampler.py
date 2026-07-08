# Random_Sampler Object
import random

from Client_Sampler import Client_Sampler

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Random_Sampler(Client_Sampler):
	def __init__(self, K, C, ALPHA, R, SEED):
		self.K = K # Number of devices in each cluster
		self.C = C # Number of simulated clusters
		self.num_clients_per_round =  self.K * self.C # Total number of clients per round
		self.ALPHA = ALPHA # Sampling ratio
		self.total_num_clients = round(self.num_clients_per_round / self.ALPHA) # Total number of clients
		self. R = R
		self.seed = SEED

	def get_samplers(self):
		#random.seed(self.seed)
		self.indices_samples = []
		for r in range(self.R):
			self.indices_samples.append(random.sample(range(self.total_num_clients), k = self.num_clients_per_round)) # Without replacement
  
def unit_test():
	K = 2
	C = 2
	ALPHA = 0.2
	R = 3
	
	client_sampler = Random_Sampler(K, C, ALPHA, R)
	client_sampler.get_samplers()
	print(client_sampler.indices_samles)
	print('---------------------------')
	INDEX = 1
	for r in range(R):
		clinet_ids = client_sampler.indices_samles[r*K*C:(r+1)*K*C]
		for c in range(C):
			part_index = INDEX + c * K
			print(part_index, clinet_ids[part_index])
		

if __name__ == '__main__':
	unit_test()
	