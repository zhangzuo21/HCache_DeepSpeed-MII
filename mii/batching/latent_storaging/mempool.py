import torch

class MemPool:
    def __init__(self, model_size, layer_num, token_num):
        # self.pinned_buffer = torch.empty((layer_num, token_num, model_size), device='cpu', pin_memory=True)
        self.model_size = model_size
        self.layer_num = layer_num

    

        