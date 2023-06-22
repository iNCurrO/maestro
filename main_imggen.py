import pydicom
import torch
from forwardprojector.FBP import FBP
from config import get_config
from PIL import Image
import os
import numpy as np
from customlib.chores import save_image
import time
import glob
import tqdm
import torch.nn.functional as F

args = get_config()
device = torch.device(args.device)
args.num_split = 1
# args.noise=1e6
# binning_size = (4, 1)
# args.num_det = int(args.num_det / binning_size[0])
# args.det_interval *= binning_size[0]
# args.recon_interval = args.recon_interval * binning_size[0]
# args.recon_size = [int(args.recon_size[0] / binning_size[0]), int(args.recon_size[1] / binning_size[0])]
args.no_mask = True


def main():
    FBP_model = FBP(args)
    save_mode = 'val'
    save_path = os.path.join(args.datadir, args.dataname)
    targetsino_list = glob.glob(os.path.join(save_path, save_mode, '*'))
    os.makedirs(os.path.join(save_path, 'reconimage'), exist_ok=True)
    print(f'save path is {save_path}')
    for i in targetsino_list:
        if os.path.splitext(i)[1] == '.npy':
            print(f'Loading... {i}')
            total_sinogram_np = np.load(i)
            total_sinogram = torch.FloatTensor(total_sinogram_np).reshape((1, 1, -1, total_sinogram_np.shape[1])).to(torch.device(args.device))
            recon_img = FBP_model(total_sinogram)
            recon_img = recon_img.squeeze().cpu().numpy()
            np.save(os.path.join(os.path.join(save_path, 'reconimage'), os.path.basename(i)[:-4]+'.npy'), recon_img)
            save_image(recon_img, os.path.join(os.path.join(save_path, 'reconimage'), os.path.basename(i)[:-4]+'.png'), sino=False)


if __name__ == '__main__':
    main()
    print("Jobs Done")
