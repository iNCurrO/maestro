import os
import torch
import torch.nn as nn
import numpy as np
from forwardprojector import utility

class FBP(nn.Module):
    def __init__(self, args):  # This part requires about 2GB GPU memory
        super().__init__()

        self.args = args
        if self.args.datatype == 'float':
            self.args.datatype = torch.float
        elif self.args.datatype == 'double':
            self.args.datatype = torch.double

        # angle setting
        self.rot_dir = 1  # 1 for counterclockwise -1 for clockwise
        self.rot_deg = 360
        self.d_beta = np.pi * self.rot_deg / self.args.view / 180  # angular step size in radian
        self.beta = (self.rot_dir * torch.linspace(0, (self.args.view - 1) * self.d_beta, self.args.view,
                                                   dtype=self.args.datatype))

        # Detector quarter offset mode
        self.shift = 0
        if self.args.quarter_offset:
            self.shift = 0.25

            # Filters
        self.args.Npow2 = int(2 ** (np.ceil(np.log2(2 * args.num_det - 1))))
        self.window = utility.gen_window(self.args)  # [rect, hann] (1, Npow2)
        self.recon_filter = utility.gen_filter(self.args)  # [ram-lak, cosine, shepp-logan] (1, Npow2)

        # Detector Geometry
        # Parallel Beam mode is not supported
        if self.args.geometry == 'fan':
            if self.args.mode == 'equally_spaced':
                self.delta_s = self.args.SCD / self.args.SDD * self.args.det_interval
                self.s_range = self.args.SCD / self.args.SDD * (self.args.det_interval *
                                                                torch.linspace((1 - self.args.num_det) / 2,
                                                                               (self.args.num_det - 1) / 2,
                                                                               self.args.num_det,
                                                                               dtype=self.args.datatype) + self.shift)
                gamma = torch.atan2(self.s_range, torch.Tensor([self.args.SCD]))
                self.delta_beta = (self.beta[-1] + self.beta[1] - self.beta[0]) / self.args.view
                self.cos_weight = torch.cos(gamma)
            elif self.args.mode == 'equiangular':
                y_center = (self.args.num_det + 1) / 2 - self.shift / self.args.det_interval
                self.delta_s = self.args.det_interval / self.args.SDD
                self.s_range = 1 / self.args.SDD * (self.args.det_interval *
                                                    torch.linspace((1 - self.args.num_det) / 2,
                                                                   (self.args.num_det - 1) / 2, self.args.num_det,
                                                                   dtype=self.args.datatype) + self.shift)
                gamma = (torch.arange(1, args.num_det + 1,
                                      dtype=self.args.datatype) - y_center) * self.delta_s  # gamma = s_range in this geometry
                self.delta_beta = 2 * np.pi / self.args.view
                self.cos_weight = self.args.SCD * torch.cos(gamma)

        # Projection geometry setting
        self.square, self.grid = self.param_setting()
        self.mask = self.gen_mask()
        self.recon_filter, self.window, self.cos_weight, self.square, self.grid, self.mask = \
            self.prepare(self.recon_filter, self.window, self.cos_weight, self.square, self.grid, self.mask)

    def param_setting(self):
        x = self.args.recon_interval * torch.linspace((1 - self.args.recon_size[0]) / 2,
                                                      (self.args.recon_size[0] - 1) / 2,
                                                      self.args.recon_size[0],
                                                      dtype=self.args.datatype) + self.args.ROIx
        y = self.args.recon_interval * torch.linspace((1 - self.args.recon_size[1]) / 2,
                                                      (self.args.recon_size[1] - 1) / 2,
                                                      self.args.recon_size[1],
                                                      dtype=self.args.datatype) + self.args.ROIy
        x_mat, y_mat = torch.meshgrid(x, y, indexing='xy')
        r = torch.sqrt(torch.pow(x_mat, 2) + torch.pow(y_mat, 2))  # (x,y,2)
        phi = torch.atan2(y_mat, x_mat)  # (x,y)
        phi[torch.isnan(phi)] = 0
        if self.args.geometry == 'fan':
            if self.args.mode == 'equally_spaced':
                U = (self.args.SCD + r[None, :, :] * torch.sin(
                    self.beta[:, None, None] - phi[None, :, :])) / self.args.SCD  # (view,x,y)
                s_xy = r[None, :, :] * torch.cos(
                    self.beta[:, None, None] - phi[None, :, :]) / U  # s value of each coord (view,x,y)
                s_xy_n = torch.reshape(s_xy / self.s_range[-1],
                                       [self.args.view, -1])  # convert range as [-1 1], (view,x*y)
                view_xy_n = torch.tile(
                    torch.linspace(-1, 1, int(self.args.view / self.args.num_split), dtype=self.args.datatype)[:, None],
                    [self.args.num_split, self.args.recon_size[0] * self.args.recon_size[1]])
                grid = torch.stack([s_xy_n, view_xy_n], dim=2)  # (view,x*y,2)
                square = torch.reshape(torch.pow(U, 2), [self.args.view, -1])  # (view,x*y)
                self.s_xy_n = s_xy_n
                return square, grid

            elif self.args.mode == 'equiangular':
                L = torch.sqrt(
                    torch.pow(self.args.SCD + r[None, :, :] * torch.sin(self.beta[:, None, None] - phi[None, :, :]), 2)
                    + torch.pow(- r[None, :, :] * torch.cos(self.beta[:, None, None] - phi[None, :, :]), 2))
                s_xy = torch.atan2(r[None, :, :] * torch.cos(self.beta[:, None, None] - phi[None, :, :]),
                                   self.args.SCD + r * torch.sin(
                                       self.beta[:, None, None] - phi[None, :, :]))  # s value of each coord (view,x,y)
                s_xy_n = torch.reshape(s_xy / self.s_range[-1],
                                       [self.args.view, -1])  # convert range as [-1 1], (view,x*y)
                view_xy_n = torch.tile(
                    torch.linspace(-1, 1, int(self.args.view / self.args.num_split), dtype=self.args.datatype)[:, None],
                    [self.args.num_split, self.args.recon_size[0] * self.args.recon_size[1]])
                grid = torch.stack([s_xy_n, view_xy_n], dim=2)  # (view,x*y,2)
                square = torch.reshape(torch.pow(L, 2), [self.args.view, -1])  # (view, x*y)
                return square, grid

    def forward(self, sinogram):
        batch, ch, view, N = sinogram.shape
        sinogram = torch.fft.fft(
            sinogram * self.cos_weight[None, None, None, :],
            n=self.args.Npow2, dim=-1
        ) * self.recon_filter[None, None, :, :] * self.window[None, None, :, :]
        sinogram = torch.real(self.delta_s * torch.fft.ifft(sinogram, dim=-1))
        sinogram = sinogram[:, :, :, N - 1:2 * N - 1]  # cut zero-padded components

        recon_img = torch.nn.functional.grid_sample(sinogram, torch.tile(self.grid, [batch, 1, 1, 1]),
                                                    mode='nearest', padding_mode='zeros',
                                                    align_corners=False)  # (batch, ch, view, x*y)
        recon_img = torch.sum(recon_img / self.square, dim=-2)  # (batch, ch, x*y)
        recon_img = self.d_beta * recon_img[:, :, :, None] \
            .view(batch, -1, self.args.recon_size[0], self.args.recon_size[1])  # [batch,ch,x,y]
        if not self.args.no_mask:
            recon_img *= self.mask[None, None, :, :]
        return recon_img

    def prepare(self, *args):
        def _prepare(tensor):
            return tensor.to(torch.device(self.args.device))

        if len(args) <= 1:
            return args[0].to(torch.device(self.args.device))
        else:
            return [_prepare(a) for a in args]

    def gen_mask(self):
        det_edge = self.args.det_interval * (self.args.num_det - 1) / 2
        if self.args.geometry == 'fan':
            if self.args.mode == 'equally_spaced':
                gamma = torch.atan(det_edge, self.args.SDD)
            elif self.args.mode == 'equiangular':
                gamma = torch.Tensor([self.args.det_interval * (self.args.num_det - 1) / 2 / self.args.SDD])
        radius = self.args.SCD * torch.sin(gamma) / self.args.recon_interval
        mask = torch.zeros(self.args.recon_size[0], self.args.recon_size[1])
        x = torch.linspace(-(self.args.recon_size[0] - 1) / 2, (self.args.recon_size[0] - 1) / 2,
                           self.args.recon_size[0])
        y = torch.linspace(-(self.args.recon_size[1] - 1) / 2, (self.args.recon_size[1] - 1) / 2,
                           self.args.recon_size[1])
        x_mat, y_mat = torch.meshgrid(x, y, indexing='xy')
        mask = torch.pow(torch.pow(x_mat / radius, 2) + torch.pow(y_mat / radius, 2), 2) < 1
        return mask
