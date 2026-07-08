# We demonstrate the adaptive offloading by a simple fixed policy agent since our testbed is easy to solve.
# For more complicated RL agent implementation, please refer to our FedAdapt paper.
class Agent:
    def __init__(self, fedadapt_config):
        self.fedadapt_config = fedadapt_config
    
    def pre_profiling(self, rep_client_ip):
        # A simple fixed policy that assign pp = -1 for the fastest group and pp = 1 for others
        self.optim_pps = {}   
        fastest_group_id = None
        for group_id in rep_client_ip:
            train_lantency_per_batch = rep_client_ip[group_id][1][1]
            if fastest_group_id is None:
                fastest_group_id = group_id
            else: 
                if train_lantency_per_batch < rep_client_ip[fastest_group_id][1][1]:
                    fastest_group_id = group_id

        for group_id in rep_client_ip:
            if group_id == fastest_group_id:
                self.optim_pps[group_id] = -1
            else:
                self.optim_pps[group_id] = 1
    
    def action(self, group_id):
        return self.optim_pps[group_id]


    