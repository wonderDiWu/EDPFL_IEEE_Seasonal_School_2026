import torch

class Layer_Regularizer:
    def __init__(self, fedfreeze_config):
        self.fedfreeze_config = fedfreeze_config
        self.mu = fedfreeze_config['Layer_Regularizer']['mu']

    def build(self):
        self.adaptive_mu_factor = 1
        
    def generate_T_list(self, init_net, cur_net, T):
        G_list = []
        T_list = []
        cliped_T_list = []
        adaptive_mu = self.mu / self.adaptive_mu_factor
        print(adaptive_mu,adaptive_mu,adaptive_mu)
        for w, w_t in zip(cur_net.parameters(), init_net.parameters()):
            if len(w.size()) >= 2:
                G_list.append(torch.mean(torch.abs(w - w_t)))
        
        non_frozen_G_list = [G_i for G_i in G_list if G_i > 0]
        for G_i in G_list:
            if G_i > 0:
                T_list.append(T * min(non_frozen_G_list) / G_i)
            else: # Global Frozen layer
                T_list.append(0)
        cliped_T_list = [min(int(T_i * adaptive_mu), T) for T_i in T_list]
        return cliped_T_list
