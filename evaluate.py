import os.path

from config import get_config
from customlib.chores import *
import torch
from forwardprojector.FBP import FBP
from models import mae
from customlib.dataset import set_dataset
from customlib.metrics import *
# Parse configuration
config = get_config()


def evaluate(network, valdataloader, Amatrix, saveimg=False, savedir = None):
    total_PSNR = 0.0
    total_SSIM = 0.0
    total_MSE = 0.0
    num_data = len(valdataloader)
    for batch_idx, samples in enumerate(valdataloader):
        sino = samples
        _, denoised_sino, _ = network(sino.cuda(), num_masked_views=config.num_masked_views)
        clean_img = Amatrix(sino.cuda())

        denoised_img = Amatrix(denoised_sino)
        total_SSIM += calculate_SSIM(clean_img, denoised_img)/num_data
        total_PSNR += calculate_psnr(clean_img, denoised_img)/num_data
        total_MSE += calculate_MSE(clean_img, denoised_img).detach().item()/num_data
        if saveimg:
            save_image
            save_images(
                clean_img.cpu().detach().numpy(), 'clean', str(batch_idx), os.path.join(savedir),
                config.valbatchsize
            )
            save_images(
                denoised_img.cpu().detach().numpy(), 'denoised', str(batch_idx), os.path.join(savedir),
                config.valbatchsize
            )
        torch.cuda.empty_cache()
    return total_SSIM, total_PSNR, total_MSE

def evaluate_main(resumenum=None, __savedir__=None):
    # initialize dataset
    print(f"Data initialization: {config.dataname}\n")
    _, valdataloader, _ = set_dataset(config)

    # Initiialize model
    print(f'Network initialization: {config.mode}\n')
    network = mae.MaskedAutoEncoder(
        num_det=config.num_det,
        num_views=config.view,
        embed_dim=config.e_dim,
        depth = config.e_depth,
        num_heads=config.e_head,
        decoder_depth=config.d_depth,
        decoder_embed_dim=config.d_dim,
        decoder_num_heads=config.d_head,
        select_view=config.select_view,
        cls_token=True,
    )

    # initialize optimzier
    optimizer = set_optimizer(config, network)

    # initialize Amatrix
    print(f"Amatrix initialization...")
    Amatrix = FBP(config)
    print(f"Amatrix initialization finished!")

    # Set log
    print(f"Resume from: {config.resume}\n")

    if not os.path.exists(os.path.join(__savedir__, 'test_result')):
        os.mkdir(os.path.join(__savedir__, 'test_result'))
    __savedir__ = os.path.join(__savedir__, 'test_result')

    print(f"Evaluation logs will be archived at the {__savedir__}\n")
    resume_network(resume=resumenum, network=network, optimizer=optimizer, config=config)
    network.eval()
    total_SSIM, total_PSNR, total_MSE = evaluate(network, valdataloader, Amatrix)

    log_str = f'Finished! SSIM: {total_SSIM}, PSNR: {total_PSNR}, '\
              f'MSE in image domain: {total_MSE}, ' \
              f'For total {len(valdataloader)}'
    print(log_str)
    with open(os.path.join(__savedir__, 'validation_logs.txt'), 'w') as log_file:
        print(log_str, file=log_file)


if __name__ == "__main__":
    temp_dir = [filename for filename in os.listdir(config.logdir) if filename.startswith(f"{int(config.resume.split('-')[0]):03}")]
    assert len(temp_dir) == 1, f'Duplicated file exists or non exist: {temp_dir}'
    evaluate_main(config.resume, os.path.join(config.logdir, temp_dir[0]))
