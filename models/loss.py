import torch

class loss_engine:
    def __init__(self, network=None, mask_ratio: float = 0.75, 
                 cycle_masking=False):
        self._network = network
        self._mask_ratio = mask_ratio
        self._cycle_masking = cycle_masking # TODO
        self._loss_func = torch.nn.MSELoss()
        
    def run_masking(self, sinogram, mask=None):
        return self._network.masking(sinogram, self._mask_ratio, mask)
    
    def run_mae(self, sinogram):
        loss, pred, mask = self._network(sinogram, self._mask_ratio)
        return loss, pred, mask
    
    def accumulate_gradients(self, sinogram):
        loss, _, _ = self.run_mae(sinogram)
        loss.backward()
        return loss.item()
    


