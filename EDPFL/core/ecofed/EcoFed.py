from .initializer import Initializer
from .compressor import Compressor
from .buffer import Buffer
from .activation_switch import Activation_Switch

class EcoFed:
    def __init__(self, ecofed_config):
        self.ecofed_config = ecofed_config
        self.initializer = Initializer(ecofed_config)
        self.compressor = Compressor(ecofed_config)
        self.replay_buffer = Buffer(ecofed_config)
        self.activation_switch = Activation_Switch(ecofed_config)
