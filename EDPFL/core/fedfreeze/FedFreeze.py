import copy

from .global_freezer import Global_Freezer
from .convergence_monitor import Convergence_Monitor
from .local_freezer import Local_Freezer
from .layer_regularizer import Layer_Regularizer

class FedFreeze:
    def __init__(self, fedfreeze_config):
        self.fedfreeze_config = fedfreeze_config
        self.global_freezer = Global_Freezer(fedfreeze_config)
        self.convergence_monitor = Convergence_Monitor(fedfreeze_config)
        self.local_freezer = Local_Freezer(fedfreeze_config)
        self.layer_regularizer = Layer_Regularizer(fedfreeze_config)
        self.current_frozen_params = {} # This is used for loading the frozen params
