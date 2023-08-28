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
        if self._cycle_masking:
            with torch.autocast(device_type='cuda'):
                loss, pred, mask, mask_rm, recovered_rm = self._network(sinogram, self._num_masked_views)
            return loss, pred, mask, mask_rm, recovered_rm
        else:
            with torch.autocast(device_type='cuda'):
                loss, pred, mask, _, _ = self._network(sinogram, self._num_masked_views)
            return loss, pred, mask, None, None
    
    def accumulate_gradients(self, sinogram, accumiter = 1, scaler=None):
        loss, _, _, _, _ = self.run_mae(sinogram)
        totalloss = 0
        for ii in range(len(loss)):
            totalloss += loss[ii]/accumiter
        scaler.scale(totalloss).backward()
            # (loss[ii]/accumiter).backward()
        return [loss[ii].cpu().detach().item() for ii in range(len(loss))]
    


