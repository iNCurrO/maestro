import torch

class loss_engine:
    def __init__(self, network=None, num_masked_views: int = 18, 
                 cycle_masking=False):
        self._network = network
        self._num_masked_views = num_masked_views
        self._cycle_masking = cycle_masking # TODO
        self._loss_func = torch.nn.MSELoss()
        
    def run_masking(self, sinogram, mask=None):
        return self._network.masking(sinogram, self._num_masked_views, mask)
    
    def run_mae(self, sinogram):
        loss, pred, mask = self._network(sinogram, self._num_masked_views)
        return loss, pred, mask
    
    def accumulate_gradients(self, sinogram):
        loss, _, _ = self.run_mae(sinogram)
        for ii in range(len(loss)):
            loss[ii].backward()
        return [loss[ii].cpu().detach().item() for ii in range(len(loss))]
    


