from .preprocessor import Pre_Processor
from .clustering import Clustering
from .agent import Agent
from .postprocessor import Post_Processor

class FedAdapt:
    def __init__(self, fedadapt_config):
        self.fedadapt_config = fedadapt_config
        self.preprocessor = Pre_Processor(fedadapt_config)
        self.clustering = Clustering(fedadapt_config)
        self.agent = Agent(fedadapt_config)
        self.postprocessor = Post_Processor(fedadapt_config)
