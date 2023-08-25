import os.path
import time
import torch
import torch.cuda.amp as amp
import math

if not os.name == 'nt':
    import vessl
from customlib.chores import save_network, save_images, lprint, adjust_lr
# from customlib.metrics import * # TODO
from models.loss import *
from forwardprojector.FBP import FBP 
from evaluate import evaluate 
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
    scaler = amp.GradScaler()

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
        network=network, num_masked_views=config.num_masked_views, cycle_masking=config.remasking
    )

    # Initialize logs
    with open(os.path.join(log_dir, 'logs.txt'), 'w') as log_file:
        print("Start of Files ================== \n", file=log_file)


    # Initial data
    val_sino = next(iter(validation_set))
    val_batch_size = val_sino.shape[0]
    save_images(val_sino, epoch=config.startepoch, tag="target", savedir=log_dir, batchnum=val_batch_size, sino=True)
    if saving_recon_image:
        recon_img = FBP_module(val_sino.cuda()).cpu().numpy()
        save_images(recon_img,  epoch=config.startepoch, tag="target_recone", savedir=log_dir, batchnum=val_batch_size, sino=False)  
        _, val_recovered_sino, mask, mask_rm, recovered_rm = loss_func.run_mae(val_sino.to('cuda'))
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
        if config.remasking:
            save_images(
                recovered_rm.cpu().detach().numpy(),
                epoch=config.startepoch,
                tag="denoised_rm",
                savedir=log_dir,
                batchnum=val_batch_size,
                sino=True
            )
            save_images(
                mask_rm.reshape([1, 1, 360, 1]).cpu().detach().numpy(),
                epoch=config.startepoch,
                tag="masked_rm",
                savedir=log_dir,
                batchnum=val_batch_size,
                sino=False
            )
            recon_img_rm = FBP_module(recovered_rm.cpu().detach().cuda()).cpu().numpy()
            save_images(recon_img_rm,  epoch=config.startepoch, tag="denoised_recone_rm", savedir=log_dir, batchnum=val_batch_size, sino=False)  
    network.train()

    # Main Part
    start_time = time.time()
    lprint(
        f"Entering training at {time.localtime(start_time).tm_mon}/{time.localtime(start_time).tm_mday} "
        f"{time.localtime(start_time).tm_hour}h {time.localtime(start_time).tm_min}m "
        f"{time.localtime(start_time).tm_sec}s",
        log_dir=log_dir
    )

    for cur_epoch in range(config.startepoch, training_epoch):
        # iteration for one epcoh
        accumiter = 0
        optimizer.zero_grad()
        for batch_idx, samples in enumerate(training_set):
            sino = samples
            loss_item = loss_func.accumulate_gradients(
                sino.to('cuda'), config.accumiter, scaler
            )
            loss_log_text = f"["
            for ii in range(len(loss_item)):
                loss_log_text += f"{loss_item[ii]}"
                if not math.isfinite(loss_item[ii]):
                    lprint(
                        f"Error!: Loss is {loss_item[ii]} for {ii}th loss (at epoch {cur_epoch}), stopping training.",
                        log_dir=log_dir
                        )
                    exit(1)
            loss_log_text += f"]"
            if batch_idx % 99 == 0:
                nettime = time.time() - start_time
                realtime_epoch = (cur_epoch + ((batch_idx+config.batchsize) / len(training_set)))
                lprint(
                    f'Train Epoch: {cur_epoch:4d}/{training_epoch:4d}, Batch: {batch_idx:6d}/{len(training_set):6d}' +
                    f', mean(sec/epoch): '
                    f'{nettime / realtime_epoch:11.5f}'
                    f', loss:' + loss_log_text + 
                    f', ETA: {timedelta(seconds=(nettime / realtime_epoch * (training_epoch - realtime_epoch)) if not (cur_epoch==0 and batch_idx==0) else 0)}',
                    log_dir=log_dir
                )
            if (accumiter+1)%config.accumiter == 0 :
                scaler.step(optimizer)
                scaler.update()
                # optimizer.step()
                optimizer.zero_grad()
            accumiter += 1
        if config.remasking:
            vessl.log(step=cur_epoch, payload={
                "mse_loss_first": loss_item[0],
                "mse_loss_second": loss_item[1]
            })
        else:
            vessl.log(step=cur_epoch, payload={"mse_loss": loss_item[0]})

        # Save check point and evaluate
        network.eval()
        with torch.no_grad():
            if cur_epoch%10 == 0:
                val_loss, val_recovered_sino, mask, mask_rm, recovered_rm = loss_func.run_mae(val_sino.to('cuda'))
                val_loss_log_text = "["
                for ii in range(len(val_loss)):
                    val_loss_log_text += str(val_loss[ii].cpu().detach().item())
                val_ssim, val_psnr, val_mse = evaluate(network, validation_set, FBP_module)
                # Print log
                lprint(
                    f'==========================================================================\n' +
                    f'Evaluation for Epoch: {cur_epoch}/{training_epoch},' +
                    f'mean(sec/Epoch): {(time.time() - start_time) / (cur_epoch+1)}, loss:' +
                    val_loss_log_text + '\n'
                    f'metrics: SSIM [{val_ssim}], '
                    f'PSRN [{val_psnr}], '
                    f'MSE: [{val_mse}], ',
                    log_dir=log_dir
                )
                save_images(
                    val_recovered_sino.cpu().detach().numpy(),
                    epoch=cur_epoch,
                    tag="denoised",
                    savedir=log_dir,
                    batchnum=val_batch_size,
                    sino=True
                )
                if config.remasking:
                    save_images(
                        recovered_rm.cpu().detach().numpy(),
                        epoch=cur_epoch,
                        tag="denoised_rm",
                        savedir=log_dir,
                        batchnum=val_batch_size,
                        sino=True
                    )
                if config.select_view == "random":
                    save_images(
                        mask.reshape([1, 1, 360, 1]).cpu().detach().numpy(),
                        epoch=cur_epoch,
                        tag="masked",
                        savedir=log_dir,
                        batchnum=val_batch_size,
                        sino=False
                    )
                    if config.remasking:
                        save_images(
                            mask_rm.reshape([1, 1, 360, 1]).cpu().detach().numpy(),
                            epoch=cur_epoch,
                            tag="masked_rm",
                            savedir=log_dir,
                            batchnum=val_batch_size,
                            sino=False
                        )
                if saving_recon_image:
                    recon_img = FBP_module(val_recovered_sino.cpu().detach().cuda()).cpu().numpy()
                    save_images(recon_img,  epoch=cur_epoch, tag="denoised_recone", savedir=log_dir, batchnum=val_batch_size, sino=False)  
                    if config.remasking:
                        recon_img_rm = FBP_module(recovered_rm.cpu().detach().cuda()).cpu().numpy()
                        save_images(recon_img_rm,  epoch=cur_epoch, tag="denoised_recone_rm", savedir=log_dir, batchnum=val_batch_size, sino=False)  
                        
                if not os.name == 'nt':
                    vessl.log(step=cur_epoch, payload={
                        "SSIM": val_ssim,
                        "PSNR": val_psnr,
                    })
                    if config.remasking:
                        vessl.log(step=cur_epoch, payload={
                            "val_mse_loss_0": loss_item[0],
                            "val_mse_loss_1": loss_item[1]
                        })
                    else:
                        vessl.log(step=cur_epoch, payload={"val_mse_loss": loss_item[0]})
                if not os.name == 'nt':
                    vessl.log(payload={"denoised_images": [
                        vessl.Image(
                            data=val_recovered_sino.cpu().detach().numpy(),
                            caption=f'Epoch:{cur_epoch:04}'
                        )
                    ]})
                    if saving_recon_image:
                        vessl.log(payload={"denoised_recon_images": [
                            vessl.Image(
                                data=recon_img,
                                caption=f'Epoch:{cur_epoch:04}'
                            )
                        ]})
            if cur_epoch == training_epoch - 1:
                save_network(network=network, epoch=training_epoch, optimizer=optimizer, savedir=log_dir)
            elif cur_epoch != 0 and cur_epoch % checkpoint_intvl == 0:
                save_network(network=network, epoch=cur_epoch, optimizer=optimizer, savedir=log_dir)
        network.train()
        adjust_lr(optimizer, cur_epoch, config)
        if not os.name == 'nt':
            vessl.progress((cur_epoch+1)/training_epoch)

    # End Training. Close everything
    with open(os.path.join(log_dir, 'logs.txt'), 'a') as log_file:
        print(f"Training Completed: EOF", file=log_file)