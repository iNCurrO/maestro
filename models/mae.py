import torch
from timm.models.vision_transformer import Block
from models.customModule import PatchEmbed, get_1d_sincos_pos_embed
from customlib.chores import patchify, unpatchify
import math


class MaskedAutoEncoder(torch.nn.Module):
    def __init__(self, num_det=724, num_views=720, embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16, mlp_rato=4.,
                 norm_layer=torch.nn.LayerNorm, norm_pix_loss=False, pos_encoding = True, 
                 select_view = "random", cls_token=False, remasking=False) -> None:
        super().__init__()
        self._num_det = num_det
        self._num_views = num_views
        self._pos_encoding = pos_encoding
        self._cls_token = cls_token
        self._select_view = select_view
        self._remasking = remasking
        ## -------------------------------------------------------------
        # Encoder implementation
        #
        self.patch_embed = PatchEmbed(
            num_det=num_det,
            num_views=num_views,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            bias=True
            )
        
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, num_views, embed_dim), requires_grad=not pos_encoding)
        if self._cls_token:    
            self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed_cls = torch.nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=not pos_encoding)
        else:
            self.cls_token = None
            self.pos_embed_cls = None
        
        self.blocks = torch.nn.ModuleList([
            Block(embed_dim, num_heads, mlp_rato, True, norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        ## ------------------ End Of Econdoer ----------------------------
        
        ## ----------------------------------------------------------------
        # MAE decoder implementation
        #
        
        self.decoder_embed = torch.nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = torch.nn.Parameter(torch.zeros(1, num_views, decoder_embed_dim), requires_grad=not pos_encoding)
        if self._cls_token:
            self.decoder_pos_embed_cls = torch.nn.Parameter(torch.zeros(1, 1, decoder_embed_dim), requires_grad= not pos_encoding)
        else:
            self.decoder_pos_embed_cls = None
                
        self.decoder_blocks = torch.nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_rato, qkv_bias=True, norm_layer=norm_layer) for i in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = torch.nn.Linear(decoder_embed_dim, num_det, bias=True)
        ## ------------------- End Of Decoder ------------------------------
        
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()
        
    def initialize_weights(self):
        ## Function to initialize weights for both of decoder and encoder
        
        # position encoding ?
        if self._pos_encoding:
            pos_embed, pos_embed_cls = get_1d_sincos_pos_embed(
                self.pos_embed.shape[-1], self._num_views, cls_token=self._cls_token
                )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            if self._cls_token:
                self.pos_embed_cls.data.copy_(torch.from_numpy(pos_embed_cls).float().unsqueeze(0))
            
            decoder_pos_embed, decoder_pos_embed_cls = get_1d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], self._num_views, cls_token=self._cls_token
                )
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
            if self._cls_token:
                self.decoder_pos_embed_cls.data.copy_(torch.from_numpy(decoder_pos_embed_cls).float().unsqueeze(0))
        
        # init flatten functions
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        
        # init cls token (if applied) and mask token
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if self._cls_token:
            torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        
        # init others
        self.apply(self._init_weights)
        
    def random_masking(self, sinogram, num_masked_views=18):
        N, V, L = sinogram.shape
        noise = torch.rand(N, V, device=sinogram.device)
        
        # sort noise for each sample
        idx_shuffle = torch.argsort(noise, dim=1) # which elements is nth order?
        idx_restore = torch.argsort(idx_shuffle, dim=1) # numbering for each elements
        
        idx_keep = idx_shuffle[:, :num_masked_views]
        sinogram_masked = torch.gather(
            sinogram, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, L)
            )
        
        # Generate binarized mask
        mask = torch.ones([N, V], device=sinogram.device, dtype=torch.int64)
        mask[:, :num_masked_views] = 0
        mask = torch.gather(mask, dim=1, index=idx_restore) 
        
        return sinogram_masked, mask, idx_restore
    
    
    def sparse_masking(self, sinogram, num_masked_views=18):
        N, V, L = sinogram.shape
        assert V % num_masked_views == 0, print(f"The number of views ({V}) for sinogram must be divided by num_masked_views ({num_masked_views})")
        
        mask = torch.ones([N, V], device=sinogram.device, dtype=torch.int64)
        mask = mask.reshape([N, num_masked_views, int(V/num_masked_views)])
        mask[:, :, -1] = 0
        mask = mask.reshape([N, V])
        
        idx_shuffle = torch.argsort(mask, dim=1, descending=True)
        idx_restore = torch.argsort(idx_shuffle, dim=1)
        idx_keep = idx_shuffle[:, :num_masked_views]
        sinogram_masked = torch.gather(
            sinogram, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, L)
            )
        
        # Generate binarized mask
        return sinogram_masked, mask, idx_restore
        
        
    def limited_masking(self, sinogram, num_masked_views=18):
        N, V, L = sinogram.shape
        
        mask = torch.ones([N, V], device=sinogram.device, dtype=torch.int64)
        mask[:,:num_masked_views] = 0
        
        idx_shuffle = torch.argsort(mask, dim=1, descending=True)
        idx_restore = torch.argsort(idx_shuffle, dim=1)
        idx_keep = idx_shuffle[:, :num_masked_views]
        sinogram_masked = torch.gather(
            sinogram, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, L)
            )
        
        # Generate binarized mask
        return sinogram_masked, mask, idx_restore
    
    def fixed_masking(self, sinogram, mask, num_masked_views):
        N, V, L = sinogram.shape
        mask_rm = torch.ones_like(mask)
        mask_rm = mask_rm - mask
        
        idx_shuffle = torch.argsort(mask, dim=1, descending=True)
        idx_keep = idx_shuffle[:, num_masked_views:]
        sinogram_masked = torch.gather(
            sinogram, dim=1, index=idx_keep.unsqueeze(-1).repeat(1, 1, L)
            )
        
        idx_shuffle = torch.argsort(mask_rm, dim=1, descending=True)
        idx_restore_rm = torch.argsort(idx_shuffle, dim=1)
        
        # Generate binarized mask
        return sinogram_masked, mask_rm, idx_restore_rm
        
    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, torch.nn.Linear) and m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)
            
    def masking(self, x, num_masked_views=18, mask=None):
        """
        Masking function have three version:
        1. SparseView  -> Not implemented TODO
        2. Limited angle -> Not implemented TODO
        3. Random View 
        """
        if mask is None:
            if self._select_view == "random":
                return self.random_masking(x, num_masked_views)
            elif self._select_view == "sparse":
                return self.sparse_masking(x, num_masked_views)
            elif self._select_view == "limit":
                return self.limited_masking(x, num_masked_views)
            else:
                raise "Not implemented masking mode: {self._masking}"
        else:
            return self.fixed_masking(x, mask, num_masked_views)
            
    def forward_encoder(self, sinogram, num_masked_views=18):
        # embed patches
        x = self.patch_embed(sinogram)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        sinogram_masked, mask, idx_restore = self.masking(x, num_masked_views)
        
        if self._cls_token:
            # append cls token
            cls_token = self.cls_token + self.pos_embed_cls
            cls_tokens = cls_token.expand(sinogram_masked.shape[0], -1, -1)
            x = torch.cat((cls_tokens, sinogram_masked), dim=1)
        else:
            x = sinogram_masked

        # apply Transformer blocks
        for blk in self.blocks:
            latent = blk(x)
        latent = self.norm(latent)

        return latent, mask, idx_restore
    
    def re_forward_encoder(self, sinogram, mask, num_masked_views=18):
        x = self.patch_embed(sinogram)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        sinogram_masked, mask_rm, idx_restore_rm = self.masking(x, num_masked_views, mask)
        
        if self._cls_token:
            # append cls token
            cls_token = self.cls_token + self.pos_embed_cls
            cls_tokens = cls_token.expand(sinogram_masked.shape[0], -1, -1)
            x = torch.cat((cls_tokens, sinogram_masked), dim=1)
        else:
            x = sinogram_masked
            
        for blk in self.blocks:
            latent = blk(x)
        latent = self.norm(latent)
        
        return latent, mask_rm, idx_restore_rm

    def forward_decoder(self, x, idx_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        if self._cls_token:
            # append mask tokens to sequence
            mask_tokens = self.mask_token.repeat(x.shape[0], idx_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(x_, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x = x + torch.cat([self.decoder_pos_embed, self.decoder_pos_embed_cls], dim=1) 
        else:
            mask_tokens = self.mask_token.repeat(x.shape[0], idx_restore[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x, mask_tokens], dim=1)
            x_ = torch.gather(x_, dim=1, index=idx_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = torch.cat([x, x_], dim=1)
            
            # add pos embed
            x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        if self._cls_token:
            x = x[:, 1:, :]
        x = unpatchify(x)
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        # target = patchify(imgs)
        target = imgs
        # print(imgs.shape, target.shape, pred.shape)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss1 = (pred - target) ** 2
        loss2 = loss1.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss2 * mask).sum() / mask.sum()  # mean loss on removed patches
        if not math.isfinite(loss):
            print(pred, target, mask)
            print("=====================================")
            print(loss1, loss2)
            print("=====================================")
            print(loss, mask.sum())
        return loss

    def recover_image(self, pred, sinogram, mask):
        x = torch.cat([sinogram, pred], dim=1)
        x = torch.gather(x, dim=1, index=mask.reshape([x.shape[0], 1, mask.shape[-1], 1]).repeat(1, 1, 1, x.shape[-1]))
        return x

    @torch.autocast(device_type='cuda')
    def forward(self, sinogram, num_masked_views=18):
        if self._remasking:
            latent, mask, idx_restore = self.forward_encoder(sinogram, num_masked_views=num_masked_views)
            pred = self.forward_decoder(latent, idx_restore)
            loss = self.forward_loss(sinogram, pred, mask)
            recovered = self.recover_image(pred, sinogram, mask)
            latent_rm, mask_rm, idx_restore_rm = self.re_forward_encoder(recovered, mask, num_masked_views)
            pred_rm = self.forward_decoder(latent_rm, idx_restore_rm)
            loss_rm = self.forward_loss(recovered, pred_rm, mask_rm)
            recovered_rm = self.recover_image(pred, recovered, mask_rm)
            return [loss, loss_rm], recovered, mask, mask_rm, recovered_rm
        else:
            latent, mask, idx_restore = self.forward_encoder(sinogram, num_masked_views=num_masked_views)
            pred = self.forward_decoder(latent, idx_restore)
            loss = self.forward_loss(sinogram, pred, mask)
            recovered = self.recover_image(pred, sinogram, mask)
            return [loss], recovered, mask, None, None