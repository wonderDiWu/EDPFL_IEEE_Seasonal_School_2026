class Activation_Switch:
    def __init__(self, ecofed_config):
        self.method = ecofed_config['Activation_Switch']['method']
        if self.method == 'periodic':
            self.period = ecofed_config['Activation_Switch']['period']
            self.round_count = 0
        else:
            raise NotImplementedError("This method has not been implemented yet.")
    
    def ask(self):
        if self.round_count % self.period == 0:
            return True
        else:
            return False