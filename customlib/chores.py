import os
import torch
import numpy as np
import platform
from PIL import Image
from timm.optim import optim_factory
import math
# if not os.name == 'nt':
    # import vessl


def patchify(sinogram):
    """
    sinogram: (N, 1, V, D)
    x: (N, V, D)
    """
    x = sinogram.squeeze(1)
    # x = torch.permute(x, [0, 2, 1])
    
    return x
    
def unpatchify(x):
    # x = torch.permute(x, [0, 2, 1])
    N, V, D = x.shape
    sinogram = x.reshape(shape=(N, 1, V, D))
    return sinogram


def set_dir(config):
    if not os.path.exists(config.logdir):
        print("Log directory is not exist, just we configured it\n")
        os.mkdir(config.logdir)
    logdir = os.listdir(config.logdir)
    logdir.sort()
    if os.listdir(config.logdir):
        dirnum = int(logdir[-1][:3])+1
    else:
        dirnum = 0
    __savedir__ = f"{dirnum:03}_{config.masking_mode}_e_h{config.e_head}_dim{config.e_dim}_depth{config.e_depth}_d_h{config.d_head}_dim{config.d_dim}_depth{config.d_depth}"
    # if not os.name == 'nt':
    #     vessl.init(__savedir__)
    __savedir__ = os.path.join(config.logdir, __savedir__)
    if config.resume:
        print(f"Resume from: {config.resume}\n")
        __savedir__ += f"_resume{config.resume}"
    os.mkdir(__savedir__)
    return [__savedir__, dirnum]


def resume_network(resume, network, optimizer, config):
    def find_network(resume_file):
        dir_num =resume_file.split('-')[0]
        cp_num = resume_file.split('-')[1]
        try:
            logdirs = [filename for filename in os.listdir(config.logdir) if filename.startswith(f"{int(dir_num):03}")]
            if not len(logdirs) == 1:
                raise FileNotFoundError
            else:
                logdir = logdirs[0]
            fn = None
            for filename in os.listdir(os.path.join(config.logdir, logdir)):
                if filename.startswith('network-' + cp_num):
                    if fn is None:
                        fn = filename
                    else:
                        raise LookupError
            if fn is None:
                raise FileNotFoundError
            print(f'Resuming... {fn}')
            return logdir, fn
        except FileNotFoundError:
            print(f'Not founded for {resume_file}, Train with random init.\n')
            return None

    traindir, resume_file = find_network(resume_file=resume)
    if resume_file is not None:
        ckpt = torch.load(os.path.join(config.logdir, traindir, resume_file))
        network.load_state_dict(ckpt['model_state_dict'])
        network = network.cuda()
        if ckpt['optimizer_state_dict']:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        else:
            print(f'Warning! there is no optimizer save file in {resume_file}, thus optimizer is going init.\n')
        return ckpt['epoch'] 

def save_network(network, optimizer, epoch, savedir):
    snapshot_pt = os.path.join(savedir, f'network-{epoch}.pt')
    print(f"Saving network... Dir: {savedir} // Epoch: {epoch}")
    torch.save({
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, snapshot_pt)
    print(f"Save complete!")


def save_images(input_images, tag, epoch, savedir, batchnum, sino=False):
    # TODO
    nx = int(np.ceil(np.sqrt(batchnum)))
    save_image_grid(
        input_images,
        os.path.join(savedir, f'samples-{int(epoch):04}-{tag}.png'), grid_size=(nx, nx), sino=sino
    )


def save_image(img, fname, sino=False):
    # TODO
    if sino:
        lo, hi = [0, 100]
    else:
        lo, hi = [0, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    Image.fromarray(img[:, :], 'L').save(fname)


def save_image_grid(img, fname, grid_size=(1, 1), sino=False):
    # TODO
    if sino:
        lo, hi = [0, 100]
    else:
        lo, hi = [0, 1]
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    _N_diff = gw*gh - _N
    if _N_diff != 0:
        img = np.concatenate((img, np.zeros([_N_diff, C, H, W])))
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname)


def set_optimizer(config, model):
    parameters = optim_factory.param_groups_weight_decay(model, weight_decay=config.weightdecay)
    if config.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(parameters, lr=config.learningrate, betas=(0.9, 0.95), weight_decay=config.weightdecay)
    elif config.optimizer == "ADAMW":
        optimizer = torch.optim.AdamW(parameters, lr=config.learningrate, betas=(0.9, 0.95), weight_decay=config.weightdecay)
    else:
        optimizer = None
        print("Error! undefined optimizer name for this codes: {}".format(config.optimizer))
        quit()
    return optimizer


def adjust_lr(optimizer, epoch, args):
    if epoch< args.warmup_epochs:
        lr = args.learningrate * epoch/args.warmup_epochs
    else:
        lr = args.min_lr + (args.learningrate - args.min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.trainingepoch - args.warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def lprint(txt, log_dir):
    print(txt)
    with open(os.path.join(log_dir, 'logs.txt'), 'a') as log_file:
        print(txt, file=log_file)

