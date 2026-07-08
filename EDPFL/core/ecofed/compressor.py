import torch
import yaml

import utils

class Compressor:
    def __init__(self, ecofed_config):
        self.method = ecofed_config['Compressor']['Method']
        if self.method == 'Int8_Quantization':
            self.compressor = Int8_Quantization()
        else:
            raise NotImplementedError("This method has not been implemented yet.")
    
    def compress(self, tensor):
        if self.method == 'Int8_Quantization':
            return self.compressor.quant(tensor)
        else:
            raise NotImplementedError("This method has not been implemented yet.")

    def decompress(self, tensor):
        if self.method == 'Int8_Quantization':
            return self.compressor.dequant(tensor)
        else:
            raise NotImplementedError("This method has not been implemented yet.")
    
    

class Int8_Quantization:
    def __init__(self):
        self.scale, self.zero_point = 1, 0
    
    def quant(self, tensor):
        scale, zero_point = tensor.max().item() / 127, 0
        dtype = torch.qint8
        q_tensor = torch.quantize_per_tensor(tensor, scale, zero_point, dtype)
        return q_tensor

    def dequant(self, q_tensor):
        tensor = torch.dequantize(q_tensor)
        return tensor