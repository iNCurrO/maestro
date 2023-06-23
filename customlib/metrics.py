import torch
import cv2
import numpy as np
import math
from torch.nn.functional import mse_loss, conv2d


def calculate_MSE(img1, img2):
    return mse_loss(img1, img2)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1.cpu() - img2.cpu()) ** 2)
    if torch.max(img1) > 10:
        print(f"Warning: Check the maximum value of img: {torch.max(img1)}")
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_SSIM(img1, img2):
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    img1 = torch.clamp(img1, min=0.0, max=1.0).detach().cpu().numpy()
    img2 = torch.clamp(img2, min=0.0, max=1.0).detach().cpu().numpy()
    img1 = img1.astype(np.float64).squeeze()
    img2 = img2.astype(np.float64).squeeze()
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def gaussian(window_size, sigma):
    """
    Generates a list of Tensor values drawn from a gaussian distribution with standard
    diviation = sigma and sum of all elements = 1.

    Length of list = window_size
    """    
    gauss =  torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):

    # Generate an 1D tensor containing values sampled from a gaussian distribution
    # _1d_window : (window_size, 1)
    # sum of _1d_window = 1
    _1d_window = gaussian(window_size=window_size, sigma=1.5).unsqueeze(1)
    
    # Converting to 2D  : _1d_window (window_size, 1) @ _1d_window.T (1, window_size)
    # _2d_window : (window_size, window_size)
    # sum of _2d_window = 1
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
     
    # expand _2d_window to window size
    # window : (channel, 1, window_size, window_size)
    window = torch.Tensor(_2d_window.expand(channel, 1, window_size, window_size).contiguous())

    return window

def calculate_SSIM_local(img1, img2, window_size=11):

    # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),    
    L = 1
    img1 = torch.clamp(img1, min=0.0, max=1.0)
    img2 = torch.clamp(img2, min=0.0, max=1.0)
    
    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    real_size = min(window_size, height, width) 
    window = create_window(real_size, channel=channels).to(img1.device)
    
    # calculating the mu parameter (locally) for both images using a gaussian filter 
    # calculates the luminosity params
    pad = window_size//2
    mu1 = conv2d(img1, window, padding=pad, groups=channels)
    mu2 = conv2d(img2, window, padding=pad, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component 
    sigma1_sq = conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 =  conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability 
    C1 = (0.01 ) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03 ) ** 2 

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1  
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1 
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    ret = ssim_score.mean() 
    
    return ret
