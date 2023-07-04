import argparse
import config

arg_lists = []
parser = argparse.ArgumentParser()


# Config list
def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--batchsize', type=int, default=8, help='Number of batch size. Recommend power of 2')
data_arg.add_argument('--valbatchsize', type=int, default=1, help='Number of batch size. Must be square of int')
data_arg.add_argument('--datadir', type=str, default="/dataset")
data_arg.add_argument('--dataname', type=str, default="Figures_small")
data_arg.add_argument('--select_view', type=str, default="sparse", choices=["random", "sparse", "limited"])
data_arg.add_argument('--num_masked_views', type=int, default=90, help="Number of remaining views after masking procedure")

# Network
network_arg = add_argument_group('Network')
network_arg.add_argument('--masking_mode', type=str, default="mae")
network_arg.add_argument('--e_head', type=int, default=16, help='Number of heads for MSA in encoder')
network_arg.add_argument('--e_depth', type=int, default=24, help="Depth of encoder (number of stacked transformer block)")
network_arg.add_argument('--e_dim', type=int, default=1024, help="Dimension of token for encoder")
network_arg.add_argument('--d_head', type=int, default=16, help='Number of heads for MSA in decoder')
network_arg.add_argument('--d_depth', type=int, default=8, help="Depth of decoder (number of stacked transformer block)")
network_arg.add_argument('--d_dim', type=int, default=512, help="Dimension of token for decoder")

# #hyperparam
hyper_param_arg = add_argument_group('Hyperparameters')
hyper_param_arg.add_argument('--trainingepoch', type=int, default=500)
hyper_param_arg.add_argument('--optimizer', type=str, default="ADAMW", choices=["ADAM", "ADAMW"])
hyper_param_arg.add_argument('--learningrate', type=float, default=2.5e-5)
hyper_param_arg.add_argument('--min_lr', type=float, default=1e-7)
hyper_param_arg.add_argument('--weightdecay', type=float, default=1e-4)
hyper_param_arg.add_argument('--warmup_epochs', type=int, default=50)

# Forward Projection
proj_arg = add_argument_group('ForwardProejction')
proj_arg.add_argument('--img_size', type=list, default=[256, 256],
                      help='Phantom image size')
proj_arg.add_argument('--pixel_size', type=float, default=0.4525,
                      help='Pixel size of the phantom image')
proj_arg.add_argument('--quarter_offset', action='store_true', help='detector quarter offset')
proj_arg.add_argument('--geometry', type=str, default='fan', help='CT geometry')
proj_arg.add_argument('--mode', type=str, default='equiangular', help="CT detector arrangement")
proj_arg.add_argument('--noise', type=int, default=0, help='Number of photons for poisson noise. Set 0 to No noise')
proj_arg.add_argument('--view', type=int, default=60,
                      help='number of view (should be even number for quarter-offset')
proj_arg.add_argument('--num_split', type=int, default=1,
                      help='number of splitting processes for FP: fewer number guarantee faster speed by reducing number of loop but memory consuming')
proj_arg.add_argument('--datatype', type=str, default='float',
                      help='datatype of tensor: double type guarantee higher accuracy but memory consuming (double: 64bit, float: 32bit)')

# Reconstruction
recon_arg = add_argument_group('Reconstruction')
recon_arg.add_argument('--originDatasetName', type=str, default='figuredata_small')
recon_arg.add_argument('--window', type=str, default='rect', help='Reconstruction window')
recon_arg.add_argument('--cutoff', type=float, default=0.3, help='Cutoff Frequency of some windows')
recon_arg.add_argument('--ROIx', type=float, default=0, help='x ROI location')
recon_arg.add_argument('--ROIy', type=float, default=0, help='y ROI location')
recon_arg.add_argument('--recon_size', type=list, default=[512, 512], help='Reconstruction image size')
recon_arg.add_argument('--recon_filter', type=str, default='ram-lak', help='Reconstruction Filter')
recon_arg.add_argument('--recon_interval', type=float, default=0.4525, help='Pixel size of the reconstruction image')
recon_arg.add_argument('--num_interp', type=int, default=4, help='number of sinc interpolation in sinogram domain')
recon_arg.add_argument('--no_mask', action='store_true', help='Not using Masking')

# Geometry conditions
proj_arg.add_argument('--SCD', type=float, default=300, help='source-center distance (mm scale)')
proj_arg.add_argument('--SDD', type=float, default=600, help='source-detector distance (mm scale)')
proj_arg.add_argument('--num_det', type=int, default=362, help='number of detector')
proj_arg.add_argument('--det_interval', type=float, default=1, help='interval of detector (mm scale)')
proj_arg.add_argument('--det_lets', type=int, default=3, help='number of detector lets')


# System parameters
sysparm_arg = add_argument_group('System')
sysparm_arg.add_argument('--resume', type=str, default=None)
sysparm_arg.add_argument('--logdir', type=str, default='/logs')
sysparm_arg.add_argument('--numworkers', type=int, default=4)
sysparm_arg.add_argument('--training', type=bool, default=True)
sysparm_arg.add_argument('--save_intlvl', type=int, default=50)
sysparm_arg.add_argument('--debugging', type=bool, default=False)
sysparm_arg.add_argument('--hyperrecord', type=bool, default=False)
sysparm_arg.add_argument('--device', type=str, default='cuda')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config
