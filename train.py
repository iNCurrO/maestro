from config import get_config
from models import mae
from customlib.chores import *
from customlib.dataset import set_dataset
from models.training_loop import training_loop
# from evaluate import evaluate_main
import os

# if not os.name == 'nt':
#     import vessl
#     print("Initialize Vessl")
#     vessl.configure(
#             organization_name="yonsei-medisys",
#             project_name="sino-domain-loss",
#         )
#     print()


def main():
    # Parse configuration
    config = get_config()

    # initialize dataset
    print(f"Data initialization: {config.dataname}\n")
    dataloader, valdataloader, num_channels = set_dataset(config)

    # Initiialize model
    print(f'Network initialization: {config.mode}\n')
    network = mae.MaskedAutoEncoder(
        num_det=config.num_det,
        num_views=config.view,
        cls_token=True,
    )
    
    # def __init__(self, num_det=724, num_views=720, embed_dim=1024, depth=24, num_heads=16,
    #              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_rato=4.,
    #              norm_layer=torch.nn.LayerNorm, norm_pix_loss=False, pos_encoding = True, cls_token=False) -> None:

    # initialize optimzier
    optimizer = set_optimizer(config, network)

    __savedir__, __dirnum__ = set_dir(config)
    # Check Resume?
    if config.resume:
        print(f"New logs will be archived at the {__savedir__}\n")
        print("Loading.... network")
        config.startepoch = resume_network(config.resume, network, optimizer, config)
        print("loaded!")
    else:
        # Make dir
        print(f"logs will be archived at the {__savedir__}\n")
        config.startepoch = 0

    # if not os.name == 'nt':
    #     hp = {
    #         "optimizer": config.optimizer,
    #         "LR": config.learningrate,
    #         "weight_decay": config.weightdecay,
    #         "model_size": network.base_channel(),
    #         "Resume_from": config.resume
    #     }
    #     vessl.hp.update(hp)

    training_loop(
        log_dir=__savedir__,
        training_epoch=config.trainingepoch,
        checkpoint_intvl=config.save_intlvl,
        training_set=dataloader,
        validation_set=valdataloader,
        network=network,
        optimizer=optimizer,
        config=config
    )

    print(f"Train Done!")

    # evaluate_main(resumenum=str(__dirnum__)+'-'+str(config.trainingepoch), __savedir__=__savedir__)
    # print(f"Testing Done!")
    # if not os.name == 'nt':
    #     vessl.finish()


if __name__ == "__main__":
    main()
