from config import get_config
from models import mae
from customlib.chores import *
from customlib.dataset import set_dataset
from models.training_loop import training_loop
from evaluate import evaluate_main
import os

if not os.name == 'nt':
    import vessl
    print("Initialize Vessl")
    vessl.configure(
            # organization_name="yonsei-medisys",
            # project_name="maestro",
        )
    print()


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

    if not os.name == 'nt':
        hp = {
            "Maskingmode": config.masking_mode,
            "Select view": config.select_view,
            "Masked view number": config.num_masked_views,
            "LR": config.learningrate,
            "weight_decay": config.weightdecay,
            "Encoder_head": config.e_head,
            "Encoder_depth": config.e_depth,
            "Encoder_Dim": config.e_dim,
            "Decoder_head": config.d_head,
            "Decoder_depth": config.d_depth,
            "Decoder_Dim": config.d_dim
        }
        vessl.hp.update(hp)

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

    evaluate_main(resumenum=str(__dirnum__)+'-'+str(config.trainingepoch), __savedir__=__savedir__)
    print(f"Testing Done!")
    if not os.name == 'nt':
        vessl.finish()


if __name__ == "__main__":
    main()
