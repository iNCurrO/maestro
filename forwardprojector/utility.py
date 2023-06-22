import torch
import numpy as np
import gc


def gen_window(args):
    if args.window == 'rect':
        w = torch.ones(1, args.Npow2, dtype=args.datatype)
    elif args.window == 'hann':
        hanning = torch.Tensor(np.hanning(args.Npow2 + 1), dtype=args.datatype)
        w = torch.fft.fftshift(hanning[:-1])
    else:
        assert args.proj_filter, 'No such Filter name'
    return w


def ramp_filter(args):
    g = torch.zeros([1, args.num_det], dtype=args.datatype)
    g[:] = float('nan')

    if args.mode == 'equiangular':
        delta = args.det_interval / args.SDD
        g[:, 0] = 1 / (8 * delta ** 2)
        for n in range(1, args.num_det):
            if n % 2 == 1:
                g[:, n] = -0.5 / (np.pi * np.sin(n * delta)) ** 2
            else:
                g[:, n] = 0
    elif args.mode == 'equally_spaced':
        delta = args.SCD / args.SDD * args.det_interval
        g[:, 0] = 1 / (8 * delta ** 2)
        for n in range(1, args.num_det):
            if n % 2 == 1:
                g[:, n] = -0.5 / (np.pi * n * delta) ** 2
            else:
                g[:, n] = 0

    g = torch.cat([torch.fliplr(g[:, 1:]), g], axis=1)
    return g


def gen_filter(args):
    N = args.num_det
    cutoff = args.cutoff
    g = ramp_filter(args)  # tensor filter for ram-lak

    x = np.arange(0, N) - (N - 1) / 2
    w_r = 2 * np.pi * x[0:-1] / (N - 1)
    ss = w_r / (2 * cutoff)

    if args.recon_filter == 'shepp-logan':
        zero = np.where(w_r == 0)
        g[zero] = g[zero] * torch.sin(ss, dtype=args.datatype) / ss
    elif args.recon_filter == 'hamming':
        g = g * (0.54 + 0.46 * (torch.cos(w_r / cutoff, dtype=args.datatype)))
    elif args.recon_filter == 'hann':
        g = g * (0.5 + 0.5 * (torch.cos(w_r / cutoff, dtype=args.datatype)))
    elif args.recon_filter == 'cosine':
        g = g * torch.cos(ss)
    gft = torch.fft.fft(g, n=args.Npow2)

    return gft