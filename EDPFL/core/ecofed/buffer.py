class Buffer:
    def __init__(self, ecofed_config):
        self.buffer_size = ecofed_config['Buffer']['buffer_size']
        self.buffer_device = ecofed_config['Buffer']['buffer_device']
        self.buffer = {}

    def update(self, pp, id, batch_id, activations, labels):
        activations = activations.to(self.buffer_device)
        labels = labels.to(self.buffer_device)
        if self.get_buffer_size() <= self.buffer_size:
            if pp not in self.buffer.keys():
                self.buffer[pp] = {}
            if id not in self.buffer[pp].keys():
                self.buffer[pp][id] = {}
            self.buffer[pp][id][batch_id] = Buffer_Chunk(activations, labels)
        else:
            raise ValueError("Buffer Overflow")
    
    def get_buffer_size(self):
        current_buffer_size = 0
        for pp in self.buffer.keys():
            current_buffer_size += len(self.buffer[pp])
        return current_buffer_size

    
class Buffer_Chunk:
    def __init__(self, activations, labels):
        self.activations = activations
        self.labels = labels