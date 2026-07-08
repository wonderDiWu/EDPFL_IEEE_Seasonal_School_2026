import numpy as np
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, fedadapt_config):
        self.num_groups = fedadapt_config['Clustering']['num_groups']

    def fit(self, observations):
        data = np.array(list(observations.values()))
        self.kmeans = KMeans(n_clusters=self.num_groups, random_state=0).fit(data)

        rep_client_ip = {}
        for i, (client_ip, observation) in enumerate(observations.items()):
            group_id = self.kmeans.labels_[i]
            if group_id not in rep_client_ip:
                rep_client_ip[group_id] = [client_ip, observation]
            else:
                if observation > rep_client_ip[group_id][1]: # We choose the client with slowest training speed as representive client
                    rep_client_ip[group_id] = [client_ip, observation]
        return rep_client_ip

    def predict(self, observation):
        observation = np.array([observation])
        return self.kmeans.predict(observation)[0]