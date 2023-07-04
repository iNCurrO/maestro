import pydicom
import torch
from forwardprojector.FP import FP
from config import get_config
import os
import numpy as np
import time
import random
from scipy.io import loadmat
import glob
from customlib.chores import save_image
import mat73

args = get_config()

device = torch.device(args.device)
# args.noise=1e6


def main():
    print("Preparing Amatrix...")
    FP_model = FP(args)
    print("Amatrix ready!")

    targetimg_list = glob.glob(os.path.join(args.datadir, args.originDatasetName, '*'))
    targetimg_list = random.shuffle(targetimg_list)
    save_path = os.path.join(args.datadir, args.dataname)
    os.makedirs(save_path, exist_ok=True)
    print('save path is {}'.format(save_path))
    for count, i in enumerate(targetimg_list):
        if count+1<int(len(targetimg_list)*0.9):
            save_mode = "train"
        else:
            save_mode = "val"
        if not os.path.exists(os.path.join(save_path, save_mode)):
            os.mkdir(os.path.join(save_path, save_mode))
        if os.path.splitext(i)[1] in [".mat", ".dcm"]:
            print(f'Loading... {i} at {save_path}')
            # data = pydicom.dcmread(os.path.join(path, i))
            # img = data.pixel_array*data.RescaleSlope
            try:
                data = loadmat(i)
            except NotImplementedError:
                data = mat73.loadmat(i)
            img = data["ph"]
            img = torch.FloatTensor(img).unsqueeze(0).permute(3, 0, 1, 2).detach()
            tic = time.time()
            sinogram = FP_model(img.to(device)).detach()
            toc = time.time()
            sinogram = sinogram.squeeze().cpu().numpy()
            # np.save(os.path.join(save_path, save_mode, os.path.basename(i)[:-4]+'.npy'), sinogram)
            for j in range(sinogram.shape[0]):
                np.save(os.path.join(save_path, save_mode, os.path.basename(i)[:-4]+'_'+str(j)+'.npy'), sinogram[j, : ,:])
            #     save_image(sinogram[j, :, :], os.path.join(save_path, os.path.basename(i)[:-4]+str(j)+'.png'), sino=False)
            print('Done, time: {}'.format(toc-tic))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    print("Jobs Done")
