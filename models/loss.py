import torch

class loss_engine:
    def __init__(self, num_masked_views: int = 18, 
                 cycle_masking=False, num_recover_views: int = 18):
        self._num_masked_views = num_masked_views
        self._num_recover_views = num_recover_views
        self._cycle_masking = cycle_masking
        self._loss_func = torch.nn.MSELoss()
        
    def run_masking(self, network, sinogram, masking_tag=None):
        return network.masking(sinogram, self._num_recover_views, masking_tag)
    
    def run_mae(self, network,  sinogram, training: bool = False, masking_tag=None):
        if training:
            masking_views = self._num_masked_views
        else:
            masking_views = self._num_recover_views
        with torch.autocast(device_type='cuda'):
            loss, pred, mask, mask_rm, recovered_rm = network(sinogram, masking_views, masking_tag)
        return loss, pred, mask, mask_rm, recovered_rm
    
    def accumulate_gradients(self, network, sinogram, accumiter = 1, masking_tag=None, scaler=None):
        loss, _, _, _, _ = self.run_mae(network, sinogram, training=True, masking_tag=masking_tag)
        totalloss = 0
        for ii in range(len(loss)):
            totalloss += loss[ii]/(accumiter * len(loss))
            # (loss[ii]/accumiter).backward()
        # scaler.scale(totalloss).backward()
        totalloss.backward()
        return [loss[ii].cpu().detach().item() for ii in range(len(loss))]
    


