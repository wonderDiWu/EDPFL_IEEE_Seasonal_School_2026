import torch

import numpy as np
import time

class Pre_Processor:
    def __init__(self, fedadapt_config):
        self.observed_num_batches = fedadapt_config['Pre_processor']['observed_num_batches']
        self.observed_batch_size = fedadapt_config['Pre_processor']['observed_batch_size']
        self.observed_channel_size = fedadapt_config['Pre_processor']['observed_channel_size']
        self.observed_input_size = fedadapt_config['Pre_processor']['observed_input_size'] 

    def observe(self, model, device, num_classes):
        # Generate random input data
        random_input = torch.randn(self.observed_batch_size, self.observed_channel_size, self.observed_input_size, self.observed_input_size)
        labels = torch.randn(self.observed_batch_size, num_classes)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        total_time = 0
        # Run the training loop
        for _ in range(self.observed_num_batches):
            # Generate random input data for each batch
            random_input = random_input.to(device)
            labels= labels.to(device)

            start_time = time.time()

            output = model(random_input)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time

        # Calculate the average time per batch
        observation = [1, total_time / self.observed_num_batches]
        return observation