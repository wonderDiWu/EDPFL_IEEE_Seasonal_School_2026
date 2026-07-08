class Post_Processor:
    def __init__(self, fedadapt_config):
        self.fedadapt_config = fedadapt_config

    def postprocess(self, action):
        pp = action
        return pp