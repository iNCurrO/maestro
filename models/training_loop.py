import os.path
import time
import torch
import math

# if not os.name == 'nt':
#     import vessl
from customlib.chores import save_network, save_images, lprint, adjust_lr
# from customlib.metrics import * # TODO
from models.loss import *
from forwardprojector.FBP import FBP 
# from evaluate import evaluate # TODO
from datetime import timedelta

def training_loop(
        log_dir: str = "./log",  # log dir
        training_epoch: int = 50,  # The number of iteration epochs
        checkpoint_intvl: int = 5,  # Interval of saving checkpoint (in epochs)
        training_set=None,  # Dataloader of training set
        validation_set=None,  # Dataloader of validation set
        network=None,  # Constructed network
        optimizer=None,  # Used optimizer
        saving_recon_image=True,
        config=None,
):
    device = torch.device(config.device)
    network = network.to(device)

    # Print parameters
    print()
    # TODO
    print()

    # Generate Amatrix
    if saving_recon_image:
        print(f"Amatrix initialization...")
        # Amatrix = FP(config)
        FBP_module = FBP(config)
        print(f"Amatrix initialization finished!")

    # Constructing losses
    print("Loading loss function....")
    loss_func = loss_engine(
        network=network,
    )

    # Initialize logs
    with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
        print("Start of Files ================== \n", file=log_file)


    # Initial data
    val_sino = next(iter(validation_set))
    val_batch_size = val_sino.shape[0]
    save_images(val_sino, epoch=0, tag="target", savedir=log_dir, batchnum=val_batch_size, sino=True)
    if saving_recon_image:
        recon_img = FBP_module(val_sino.cuda()).cpu().numpy()
        save_images(recon_img,  epoch=0, tag="target_recone", savedir=log_dir, batchnum=val_batch_size, sino=False)  
        if config.resume:
            val_loss, val_recovered_sino, mask = loss_func.run_mae(val_sino.to('cuda'))
            recon_img = FBP_module(val_recovered_sino.cpu().detach().cuda()).cpu().numpy()
            save_images(
                val_recovered_sino.cpu().detach().numpy(),
                epoch=config.startepoch,
                tag="denoised",
                savedir=log_dir,
                batchnum=val_batch_size,
                sino=True
            )
            save_images(
                mask.reshape([1, 1, 360, 1]).cpu().detach().numpy(),
                epoch=config.startepoch,
                tag="masked",
                savedir=log_dir,
                batchnum=val_batch_size,
                sino=False
            )
            save_images(recon_img,  epoch=config.startepoch, tag="denoised_recone", savedir=log_dir, batchnum=val_batch_size, sino=False)  
    network.train()

    # Main Part
    start_time = time.time()
    print(
        f"Entering training at {time.localtime(start_time).tm_mon}/{time.localtime(start_time).tm_mday} "
        f"{time.localtime(start_time).tm_hour}h {time.localtime(start_time).tm_min}m "
        f"{time.localtime(start_time).tm_sec}s"
    )

    for cur_epoch in range(config.startepoch, training_epoch):
        # iteration for one epcoh
        logs = ""
        for batch_idx, samples in enumerate(training_set):
            optimizer.zero_grad()
            sino = samples # TODO
            loss_item = loss_func.accumulate_gradients(
                sino.to('cuda')
            )
            if batch_idx % 99 == 0:
                nettime = time.time() - start_time
                realtime_epoch = (cur_epoch + ((batch_idx+config.batchsize) / len(training_set)))
                print(
                    f'Train Epoch: {cur_epoch}/{training_epoch}, Batch: {batch_idx}/{len(training_set)}' +
                    f'mean(sec/epoch): '
                    f'{nettime / realtime_epoch}'
                    f', loss:' +
                    str(loss_item) +
                    f'ETA: {timedelta(seconds=(nettime / realtime_epoch * (training_epoch - realtime_epoch)) if not (cur_epoch==0 and batch_idx==0) else 0)}'
                )
            if not math.isfinite(loss_item):
                print(f"Error!: Loss is {loss_item} (at epoch {cur_epoch}), stopping training.")
                exit(1)
            optimizer.step()
        # vessl.log(step=cur_epoch, payload={"trainingloss_"+keys: logs[keys] for keys in logs})

        # Save check point and evaluate
        network.eval()
        with torch.no_grad():
            val_loss, val_recovered_sino, mask = loss_func.run_mae(val_sino.to('cuda'))
            if cur_epoch%10 == 0:
                # val_ssim, val_psnr, val_mse, val_sinomse = evaluate(network, validation_set, Amatrix)
                # Print log
                print(
                    f'==========================================================================\n' +
                    f'Evaluation for Epoch: {cur_epoch}/{training_epoch},' +
                    f'mean(sec/Epoch): {(time.time() - start_time) / (cur_epoch+1)}, loss:' +
                    str(val_loss) + '\n'
                    # f'metrics: SSIM [{val_ssim}], '
                    # f'PSRN [{val_psnr}], '
                    # f'MSE: [{val_mse}], '
                    # f'sinoMSE: [{val_sinomse}]',
                )
                save_images(
                    val_recovered_sino.cpu().detach().numpy(),
                    epoch=cur_epoch,
                    tag="denoised",
                    savedir=log_dir,
                    batchnum=val_batch_size,
                    sino=True
                )
                save_images(
                    mask.reshape([1, 1, 360, 1]).cpu().detach().numpy(),
                    epoch=cur_epoch,
                    tag="masked",
                    savedir=log_dir,
                    batchnum=val_batch_size,
                    sino=False
                )
                if saving_recon_image:
                    recon_img = FBP_module(val_recovered_sino.cpu().detach().cuda()).cpu().numpy()
                    save_images(recon_img,  epoch=cur_epoch, tag="denoised_recone", savedir=log_dir, batchnum=val_batch_size, sino=False)  
                # if not os.name == 'nt':
                #     vessl.log(step=cur_epoch, payload={
                #         "SSIM": val_ssim,
                #         "PSNR": val_psnr,
                #         "MSE": val_mse,
                #         "sinoMSE": val_sinomse,
                #     })

                # if not os.name == 'nt':
                #     vessl.log(payload={"denoised_images": [
                #         vessl.Image(
                #             data=val_denoised_img.cpu().detach().numpy(),
                #             caption=f'Epoch:{cur_epoch:4}'
                #         )
                #     ]})
            if cur_epoch == training_epoch - 1:
                save_network(network=network, epoch=training_epoch, optimizer=optimizer, savedir=log_dir)
            elif cur_epoch != 0 and cur_epoch % checkpoint_intvl == 0:
                save_network(network=network, epoch=cur_epoch, optimizer=optimizer, savedir=log_dir)
        network.train()
        adjust_lr(optimizer, cur_epoch, config)
        # if not os.name == 'nt':
        #     vessl.progress((cur_epoch+1)/training_epoch)

    # End Training. Close everything
    with open(os.path.join(log_dir, 'logs.txt'), 'a') as log_file:
        print(f"Training Completed: EOF", file=log_file)