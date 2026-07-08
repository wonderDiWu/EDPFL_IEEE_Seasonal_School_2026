class Local_Freezer:
    def __init__(self, fedfreeze_config):
        self.fedfreeze_config = fedfreeze_config
        self.epsilon = fedfreeze_config['Local_Freezer']['epsilon']

    def build(self, T):
        self.T = T
        self.T_list = []
        self.cur_index = 0

    def ask(self, cur_T):
        if cur_T > self.epsilon * self.T:
            return True
        else:
            return False

    def local_freezing(self, cur_T, net):
        if cur_T > self.T_list[self.cur_index]:
            count = 0
            for name, param in net.named_parameters():
                if len(param.size()) >= 2:
                    if count > self.cur_index:
                        return
                    else:
                        param.requires_grad = False
                        count += 1
                else:
                    param.requires_grad = False
            self.cur_index += 1                  