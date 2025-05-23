# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Delete tensors after usage to free GPU memory
# - Add HRDA debug visualizations
# - Support ImageNet feature distance for LR and HR predictions of HRDA
# - Add masked image consistency
# - Update debug image system
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.
# A copy of the license is available at resources/license_dacs

import math
import os
import random
from copy import deepcopy

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, HRDAEncoderDecoder, build_segmentor
from mmseg.models.segmentors.hrda_encoder_decoder import crop
from mmseg.models.uda.masking_consistency_module import \
    MaskingConsistencyModule
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import prepare_debug_out, subplotimg
from mmseg.utils.utils import downscale_label_ratio
from plot import save_segmentation_map
from mmseg.models.segmentors.base import UNet
import torch.nn as nn
from matplotlib.colors import ListedColormap
from mmseg.datasets import CityscapesDataset
import json
from mmseg.models.uda.refinement import EncodeDecode
from mmseg.models.uda.swinir_backbone import MGDNRefinement


def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0
        self.max_iters = cfg['max_iters']
        self.source_only = cfg['source_only']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.mask_mode = cfg['mask_mode']
        self.enable_masking = self.mask_mode is not None
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        if not self.source_only:
            self.ema_model = build_segmentor(ema_cfg)
        self.mic = None
        if self.enable_masking:
            self.mic = MaskingConsistencyModule(require_teacher=False, cfg=cfg)
        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        #Adding for our training
        self.network = None
        self.optimizer = None
        self.masked_loss_list = []
        self.refin_loss_list = []

        # getting the type of refinement
        self.attention_type = cfg["attention_type"]

        with open("data/gta/sample_class_stats_dict.json","r") as of:
            self.sample_class_dict = json.load(of)

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        if self.source_only:
            return
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        if self.source_only:
            return
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs
    
    def plot_loss_evolution(self, loss_values, plot_name="loss_plot.png", out_dir="./", loss_label="Loss"):
        """
        Plots the evolution of loss over epochs.
        
        Parameters:
            loss_values (list): List of loss values recorded over epochs.
            out_dir (str): Directory to save  the loss plot.
            plot_name (str): Filename for the saved loss plot.
            loss_label (str): Label for the loss on the y-axis.
        """
        plt.figure()
        plt.plot(range(1, len(loss_values) + 1), loss_values, marker='+', linestyle='None')
        plt.xlabel('Epochs')
        plt.ylabel(loss_label)
        plt.title(f'{loss_label} Evolution')

        out_dir = os.path.join(self.train_cfg['work_dir'])
        save_path = os.path.join(out_dir, plot_name)
        plt.savefig(save_path)
        plt.close()
        print(f"Loss plot saved at {save_path}")
    
    def is_sliding_mean_loss_decreased(self, loss_list, current_iter, window_size=200):
        """
        Check if the sliding mean loss has decreased.
        
        Args:
        loss_list (list): List of loss values
        window_size (int): Size of the sliding window (default 100)
        
        Returns:
        bool: True if sliding mean loss has decreased, False otherwise
        """
        # Check if list is long enough for sliding window
        if len(loss_list) < window_size * 2 + 1 :
            return False
        
        # Calculate sliding means for first and second half of the window
        first_half_mean = sum(loss_list[current_iter-(2*window_size):current_iter-window_size]) / window_size
        second_half_mean = sum(loss_list[current_iter-window_size:current_iter]) / window_size
        
        # Return True if sliding mean has decreased
        return second_half_mean < first_half_mean
    
    def train_refinement_source(self, pl_source, sam_source, gt_source, network, optimizer, device,class_weight): #ADDED
        if network is None : #Initialization du réseau et tutti quanti
            #network = UNet() #For binary
            #network = UNet(n_classes=19) #For multilabel
            network = Refinement(pl_source.shape[-1],self.attention_type,device)
            network = network.to(device)
            optimizer = torch.optim.Adam(params=network.parameters(), lr=0.0001)
        
        network.train()
        #ce_loss = torch.nn.BCEWithLogitsLoss() #uncomment for binary
        ce_loss = nn.CrossEntropyLoss(ignore_index=255,weight=class_weight.to(device)) #For multilabel
        pl_source = pl_source.unsqueeze(1)
        #concat = torch.cat((pl_source, sam_source), dim=1).float()
        
        pred = network(pl_source,sam_source)
        print("pred_shape", pred.shape, "pred_unique", np.unique(pred.detach().cpu().numpy()))
        print("pred_shape", gt_source.shape, "pred_unique", np.unique(gt_source.detach().cpu().numpy()))
        #loss = ce_loss(pred, gt_source.float()) #uncomment for binary
        loss = ce_loss(pred, gt_source.squeeze(1).long()) #for multilabel
        optimizer.zero_grad()
        loss.backward()
        self.refin_loss_list.append(loss.item())
        optimizer.step()
        self.plot_loss_evolution(self.refin_loss_list,plot_name="refin_loss.png")

        return network, optimizer


    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        # If the mask is empty, the mean will be NaN. However, as there is
        # no connection in the compute graph to the network weights, the
        # network gradients are zero and no weight update will happen.
        # This can be verified with print_grad_magnitude.
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        # Features from multiple input scales (see HRDAEncoderDecoder)
        if isinstance(self.get_model(), HRDAEncoderDecoder) and \
                self.get_model().feature_scale in \
                self.get_model().feature_scale_all_strs:
            lay = -1
            feat = [f[lay] for f in feat]
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f[lay].detach() for f in feat_imnet]
            feat_dist = 0
            n_feat_nonzero = 0
            for s in range(len(feat_imnet)):
                if self.fdist_classes is not None:
                    fdclasses = torch.tensor(
                        self.fdist_classes, device=gt.device)
                    gt_rescaled = gt.clone()
                    if s in HRDAEncoderDecoder.last_train_crop_box:
                        gt_rescaled = crop(
                            gt_rescaled,
                            HRDAEncoderDecoder.last_train_crop_box[s])
                    scale_factor = gt_rescaled.shape[-1] // feat[s].shape[-1]
                    gt_rescaled = downscale_label_ratio(
                        gt_rescaled, scale_factor, self.fdist_scale_min_ratio,
                        self.num_classes, 255).long().detach()
                    fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses,
                                           -1)
                    fd_s = self.masked_feat_dist(feat[s], feat_imnet[s],
                                                 fdist_mask)
                    feat_dist += fd_s
                    if fd_s != 0:
                        n_feat_nonzero += 1
                    del fd_s
                    if s == 0:
                        self.debug_fdist_mask = fdist_mask
                        self.debug_gt_rescale = gt_rescaled
                else:
                    raise NotImplementedError
        else:
            with torch.no_grad():
                self.get_imnet_model().eval()
                feat_imnet = self.get_imnet_model().extract_feat(img)
                feat_imnet = [f.detach() for f in feat_imnet]
            lay = -1
            if self.fdist_classes is not None:
                fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
                scale_factor = gt.shape[-1] // feat[lay].shape[-1]
                gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                    self.fdist_scale_min_ratio,
                                                    self.num_classes,
                                                    255).long().detach()
                fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                                  fdist_mask)
                self.debug_fdist_mask = fdist_mask
                self.debug_gt_rescale = gt_rescaled
            else:
                feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    def update_debug_state(self):
        debug = self.local_iter % self.debug_img_interval == 0
        self.get_model().automatic_debug = False
        self.get_model().debug = debug
        if not self.source_only:
            self.get_ema_model().automatic_debug = False
            self.get_ema_model().debug = debug
        if self.mic is not None:
            self.mic.debug = debug

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=logits.device)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask):
        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            assert valid_pseudo_mask is None
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            assert valid_pseudo_mask is None
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    def forward_train(self,
                      img,
                      img_metas,
                      gt_semantic_seg,
                      sam_pseudo_label, #<-------- ADDED
                      target_img,
                      target_img_metas,
                      target_sam, #<-------- ADDED
                      rare_class=None,
                      valid_pseudo_mask=None):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        #plt.imshow(img.cpu().numpy()[0,0,:,:])
        #plt.show()
        #plt.imshow(target_img.cpu().numpy()[0,0,:,:])
        #plt.show()
        #plt.imshow(sam_pseudo_label.cpu().numpy()[0,0,:,:])
        #plt.show()
        #plt.imshow(target_sam.cpu().numpy()[0,0,:,:])
        #plt.show()

        # code to obtain the class weight for the current image
        filename = img_metas[0]["filename"]
        oriname = img_metas[0]["ori_filename"]
        new_name=oriname.replace(".png","_labelTrainIds.png")

        gt_filename = filename.replace(f"images/{oriname}",f'labels/{new_name}')
        gt_class_stats = self.sample_class_dict[gt_filename]
        gt_class_stats = {int(k): v for k,v in gt_class_stats.items()}

        # Invert class frequencies
        gt_class_weights = [gt_class_stats.get(i, 0) for i in range(19)]
        gt_class_weights = [1/weight if weight != 0 else 0 for weight in gt_class_weights]

        # Convert to tensor
        weights = torch.tensor(gt_class_weights)

        # Avoid zeros for normalization
        nonzero_weights = weights[weights > 0]
        min_w, max_w = nonzero_weights.min(), nonzero_weights.max()

        # Rescale to [1, 5]
        gt_class_weights = torch.where(
            weights > 0,
            1 + 4 * (weights - min_w) / (max_w - min_w),
            torch.tensor(0.0)  
        )


        palette = CityscapesDataset.PALETTE
        cityscapes_cmap = ListedColormap(np.array(palette) / 255.0)

        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training
        if self.mic is not None:
            self.mic.update_weights(self.get_model(), self.local_iter)

        self.update_debug_state()
        seg_debug = {}

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }

        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feat=True)  #Goes into the hrda_encoder_decoder forward train function
        src_feat = clean_losses.pop('features')
        seg_debug['Source'] = self.get_model().debug_output
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)
        clean_loss.backward(retain_graph=self.enable_fdist)
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg,
                                                      src_feat)
            log_vars.update(add_prefix(feat_log, 'src'))
            feat_loss.backward()
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')
        del src_feat, clean_loss
        if self.enable_fdist:
            del feat_loss

        pseudo_label, pseudo_weight = None, None
        if not self.source_only:
            # Generate pseudo-label (on target)
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits = self.get_ema_model().generate_pseudo_label(
                target_img, target_img_metas)
            seg_debug['Target'] = self.get_ema_model().debug_output

            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(
                ema_logits)
            del ema_logits
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            save_segmentation_map(pseudo_label.squeeze().detach().cpu().numpy(), os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_ema.png'))

            pseudo_weight = self.filter_valid_pseudo_region(
                pseudo_weight, valid_pseudo_mask)
            gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)


            # Generate pseudo-label (on source)
            for m in self.get_ema_model().modules():
                if isinstance(m, _DropoutNd):
                    m.training = False
                if isinstance(m, DropPath):
                    m.training = False
            ema_logits_source = self.get_ema_model().generate_pseudo_label(
                img, img_metas)
            seg_debug['Source'] = self.get_ema_model().debug_output

            pseudo_label_source, pseudo_weight_source = self.get_pseudo_label_and_weight(
                ema_logits_source)
            del ema_logits_source

            #pseudo_weight = self.filter_valid_pseudo_region(
            #    pseudo_weight, valid_pseudo_mask)
            #gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)



            #C'est l'endroit où intégrer l'idée...
            ########################## TO DO #####################################
            #L'idée, c'est d'avoir :
            # - pl_target : OK  : pseudo_label
            # - sam_target : OK : target_sam
            # - pl_source : OK ? : pseudo_label_source
            # - sam_src : OK : sam_pseudo_label

            ######################################################################

            # Idée de refinement entrainé sur gt_source/sam_source ?

            #train_refinement_source(pl_source, sam_source, gt_source, network, optimizer, device)
            #print(gt_semantic_seg.shape)
            #classes = torch.unique(gt_semantic_seg)
            #nclasses = classes.shape[0]
            #print("number of classes ?", nclasses)
            #if (self.local_iter < 7500):
            if (self.is_sliding_mean_loss_decreased(self.masked_loss_list, self.local_iter) and self.local_iter < 12500):
                self.network, self.optimizer = self.train_refinement_source(pseudo_label_source, sam_pseudo_label, gt_semantic_seg, self.network, self.optimizer, dev,gt_class_weights)

            #if (self.local_iter < 7500):
            if self.is_sliding_mean_loss_decreased(self.masked_loss_list, self.local_iter) :
                with torch.no_grad():
                    self.network.eval()
                    pseudo_label = pseudo_label.unsqueeze(1)
                    #concat = torch.cat((pseudo_label, target_sam), dim=1).float()
                    pseudo_label_ref = self.network(pseudo_label,target_sam)
                    pseudo_label = pseudo_label.squeeze(1)

                    softmax = torch.nn.Softmax(dim=1)
                    pseudo_label_ref2 = torch.argmax(softmax(pseudo_label_ref),axis=1).unsqueeze(1)

                    #plt.imshow(gt_semantic_seg[0].cpu().numpy()[0, :, :])
                    #plt.show()
                    #print("unique value", np.unique(pseudo_label.cpu().numpy()))
                    #print("shape", np.shape(pseudo_label_ref.cpu().numpy()))

                for j in range(batch_size):
                    rows, cols = 1, 5  # Increase cols to 4 for the new plot
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(3 * cols, 3 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0.05,
                            'top': 0.95,
                            'bottom': 0.05,
                            'right': 0.95,
                            'left': 0.05
                        },
                    )

                    # Plot the images
                    axs[0].imshow(target_img[j].cpu().numpy()[0, :, :])
                    axs[0].set_title('Target Image')

                    subplotimg(axs[1],pseudo_label[j].cpu().numpy()[:, :],'Pseudo Label')
                    #axs[1].imshow(pseudo_label[j].cpu().numpy()[:, :], cmap=cityscapes_cmap)
                    #axs[1].set_title('Pseudo Label')

                    axs[2].imshow(target_sam[j].cpu().numpy()[0, :, :], cmap='gray')
                    axs[2].set_title('Target SAM')

                    axs[3].imshow(pseudo_label_ref[j].cpu().numpy()[0, :, :], cmap='gray')  # New plot
                    axs[3].set_title('Pseudo Label Ref')

                    subplotimg(axs[4],pseudo_label_ref2[j].cpu().numpy()[:, :],'pl_after_post')
                    # axs[4].imshow(pseudo_label_ref2[j].cpu().numpy()[0, :, :], cmap=cityscapes_cmap)  # New plot
                    # axs[4].set_title('pl_after_post')

                    # Turn off axis for all subplots
                    for ax in axs.flat:
                        ax.axis('off')

                    # Save the figure
                    out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
                    os.makedirs(out_dir, exist_ok=True)
                    plt.savefig(
                        os.path.join(out_dir, f'{(self.local_iter + 1):06d}_{j}_new.png')
                    )
                    plt.close()

                    #For binary segmentation
                    #target_sam = target_sam.squeeze(1)  # Removes the singleton dimension
                    #pseudo_label = (pseudo_label_ref.squeeze(1)>0.5).long()
                    
                    #For multilabel segmentation
                    softmax = torch.nn.Softmax(dim=1)
                    pseudo_label = torch.argmax(softmax(pseudo_label_ref),axis=1).unsqueeze(1)
                    save_segmentation_map(pseudo_label.squeeze().detach().cpu().numpy(), os.path.join(out_dir,
                                        f'{(self.local_iter + 1):06d}_pl_raffiné.png'))

                    #Let it uncommented for both
                    pseudo_label = pseudo_label.squeeze(1)
                

            # Apply mixing
            mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
            mixed_seg_weight = pseudo_weight.clone()
            mix_masks = get_class_masks(gt_semantic_seg)

            for i in range(batch_size):
                strong_parameters['mix'] = mix_masks[i]
                mixed_img[i], mixed_lbl[i] = strong_transform(
                    strong_parameters,
                    data=torch.stack((img[i], target_img[i])),  #ici source_img et target_img
                    target=torch.stack(
                        (gt_semantic_seg[i][0], pseudo_label[i]))) #ici source_label et target_pseudo_label
                _, mixed_seg_weight[i] = strong_transform(
                    strong_parameters,
                    target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
            del gt_pixel_weight
            mixed_img = torch.cat(mixed_img)
            mixed_lbl = torch.cat(mixed_lbl)

            # Train on mixed images
            mix_losses = self.get_model().forward_train(
                mixed_img,
                img_metas,
                mixed_lbl,
                seg_weight=mixed_seg_weight,
                return_feat=False,
            )
            seg_debug['Mix'] = self.get_model().debug_output
            mix_losses = add_prefix(mix_losses, 'mix')
            mix_loss, mix_log_vars = self._parse_losses(mix_losses)
            log_vars.update(mix_log_vars)
            mix_loss.backward()

        # Masked Training
        if self.enable_masking and self.mask_mode.startswith('separate'):
            masked_loss = self.mic(self.get_model(), img, img_metas,
                                   gt_semantic_seg, target_img,
                                   target_img_metas, valid_pseudo_mask,
                                   pseudo_label, pseudo_weight)
            seg_debug.update(self.mic.debug_output)
            masked_loss = add_prefix(masked_loss, 'masked')
            masked_loss, masked_log_vars = self._parse_losses(masked_loss)
            log_vars.update(masked_log_vars)
            masked_loss.backward()
            self.masked_loss_list.append(masked_loss.item())
            self.plot_loss_evolution(self.masked_loss_list,plot_name="masked_loss.png")

        if self.local_iter % self.debug_img_interval == 0 and \
                not self.source_only:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                if mixed_lbl is not None:
                    subplotimg(
                        axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3],
                    mixed_seg_weight[j],
                    'Pseudo W.',
                    vmin=0,
                    vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()

        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'], 'debug')
            os.makedirs(out_dir, exist_ok=True)
            if seg_debug['Source'] is not None and seg_debug:
                if 'Target' in seg_debug:
                    seg_debug['Target']['Pseudo W.'] = mixed_seg_weight.cpu(
                    ).numpy()
                for j in range(batch_size):
                    cols = len(seg_debug)
                    rows = max(len(seg_debug[k]) for k in seg_debug.keys())
                    fig, axs = plt.subplots(
                        rows,
                        cols,
                        figsize=(5 * cols, 5 * rows),
                        gridspec_kw={
                            'hspace': 0.1,
                            'wspace': 0,
                            'top': 0.95,
                            'bottom': 0,
                            'right': 1,
                            'left': 0
                        },
                        squeeze=False,
                    )
                    for k1, (n1, outs) in enumerate(seg_debug.items()):
                        for k2, (n2, out) in enumerate(outs.items()):
                            subplotimg(
                                axs[k2][k1],
                                **prepare_debug_out(f'{n1} {n2}', out[j],
                                                    means, stds))
                    for ax in axs.flat:
                        ax.axis('off')
                    plt.savefig(
                        os.path.join(out_dir,
                                     f'{(self.local_iter + 1):06d}_{j}_s.png'))
                    plt.close()
                del seg_debug
        self.local_iter += 1

        return log_vars
    
class Refinement(nn.Module):
    def __init__(self, dim,attention_type:str,device):
        super().__init__()
        self.attention_type = attention_type

        if attention_type == "MGDN":
            self.autoencoder = MGDNRefinement()

        if attention_type == "encode_decode":
            self.autoencoder = EncodeDecode(device)

        if attention_type not in ("no_sam", "encode_decode","MGDN") : 
            self.attention = AttentionBlock(dim,attention_type) 
        self.unet = UNet(in_channel=1,n_classes=19)
        if self.attention_type == "convolutional_cross_attention":
            self.decode = Decoder()

    def forward(self, pl_source, sam_source):
        pl_source = pl_source.float()
        sam_source = sam_source.float()

        # Extract features
        if self.attention_type not in ("no_sam", "encode_decode","MGDN"):   
            feats = self.attention(pl_source,sam_source)

        if self.attention_type == "simple_cross_attention":
            out = feats * sam_source + (1-feats)*pl_source
            out = self.unet(out)
        elif self.attention_type == "convolutional_cross_attention":
            out = feats
            out = self.decode(out)
            out = out * sam_source + (1-out)*pl_source
            out = self.unet(out)

        if self.attention_type not in ("no_sam", "encode_decode","MGDN") : 
            out = self.unet(pl_source)
        
        if self.attention_type == "encode_decode":
            out = self.autoencoder(sam_source,pl_source)
        
        if self.attention_type == "MGDN":
            out = self.autoencoder(sam_source,pl_source)

        return out

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = nn.Sequential(
            nn.ConvTranspose2d(762, 512, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 1, kernel_size=4, stride=2, padding=1)
        )
    def forward(self,latent_space):
        
        return self.module(latent_space)

class CrossAttention(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 qkv_bias=False,
                 drop_rate=0.1
                 ):
        super().__init__()
        self.W_query = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.W_key = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)
        self.W_value = nn.Linear(embed_dim,embed_dim,bias=qkv_bias)

        self.att_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(drop_rate)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim,2*embed_dim,bias=qkv_bias),
            nn.Linear(2*embed_dim,embed_dim,bias=qkv_bias)
        )
        self.ff_dropout =  nn.Dropout(drop_rate)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)


    def compute_att(self,query,key,value):
        attention_scores:torch.Tensor = (query @ key.transpose(-2,-1)) / query.shape[-1]**0.5
        attention_scores = attention_scores.view(attention_scores.size(0),-1)
        attention_weights = attention_scores.contiguous().softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_weights = attention_weights.view_as(query).contiguous()

        attention = attention_weights @ value
        return attention
        
    
    def forward(self,
                pl_source:torch.Tensor,
                sam_source:torch.Tensor
                ):
        query:torch.Tensor = self.W_query(pl_source)
        key:torch.Tensor = self.W_key(sam_source)
        value:torch.Tensor = self.W_value(sam_source)

        attention = self.compute_att(query,key,value)
        
        attention = self.att_layer_norm(attention+pl_source)

        out = self.ff(attention)
        out = self.ff_dropout(out)

        return self.ff_layer_norm(out+attention)

class AttentionBlock(nn.Module):
    def __init__(self,
                 embed_dim:int,
                 attention_type:str,
                 qkv_bias=False,
                 drop_rate=0.1,
                 num_blocks=12):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == "simple_cross_attention":
            self.layers = nn.ModuleList([CrossAttention(embed_dim,qkv_bias,drop_rate) for _ in range(num_blocks)])
        elif attention_type == "convolutional_cross_attention":
            self.positional_embed = nn.Parameter(torch.randn((1,762,64,64)))
            self.scaling_pl_source = nn.Conv2d(1,762,kernel_size=16,stride=16)
            self.scaling_sam_source = nn.Conv2d(1,762,kernel_size=16,stride=16)

            self.sam_attention = nn.ModuleList([ConvolutionalCrossAttention() for _ in range(3)])
            self.pl_attention = nn.ModuleList([ConvolutionalCrossAttention() for _ in range(6)])
            self.layers = nn.ModuleList([ConvolutionalCrossAttention() for _ in range(6)])


    def go_forward(self,x,type):
        module = self.sam_attention if type=="sam" else self.pl_attention
        x,_ = x
        for layer in module:
            x = layer(x,x)
        return x
        

    def forward(self,pl_source:torch.Tensor,
                sam_source:torch.Tensor):
        if self.attention_type == "simple_cross_attention":
            x = pl_source
            for layer in self.layers:
                x = layer(x,sam_source)
            x = x.sigmoid()
        elif self.attention_type == "convolutional_cross_attention":
            pl_source = self.scaling_pl_source(pl_source) + self.positional_embed #(batch,embed_dim,64,64)
            sam_source = self.scaling_sam_source(sam_source) + self.positional_embed #(batch,embed_dim,64,64)
            sam_attention = self.go_forward((sam_source,sam_source),"sam")
            x = pl_source
            for layer,attention in zip(self.layers,self.pl_attention):
                x = attention(x,x) # self attention
                x = layer(x,sam_attention) # cross attention
        
        return x

class ConvolutionalCrossAttention(nn.Module):
    def __init__(self,embed_dim=762,drop_rate=0.1):
        super().__init__()

        self.embed_dim = embed_dim

        self.W_query = nn.Linear(embed_dim,embed_dim,bias=False)

        self.W_key = nn.Linear(embed_dim,embed_dim,bias=False)

        self.W_value = nn.Linear(embed_dim,embed_dim,bias=False)

        self.sam_norm = nn.LayerNorm(embed_dim)
        self.pl_norm = nn.LayerNorm(embed_dim)

        self.dropout = nn.Dropout(drop_rate)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim,embed_dim*2,bias=False),
            nn.ReLU(),
            nn.Linear(2*embed_dim,embed_dim,bias=False)
        )
        self.ff_dropout =  nn.Dropout(drop_rate)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
    
    def compute_att(self,query,key,value):
        
        attention_scores:torch.Tensor = (query @ key.transpose(-2,-1)) / query.shape[-1]**0.5
        attention_weights = attention_scores.contiguous().softmax(dim=-1)
        attention_weights = self.dropout(attention_weights)

        attention = attention_weights @ value
        return attention
    
    def forward(self,
                pl_source:torch.Tensor,
                sam_source:torch.Tensor
                ):
        batch,emed_dim,_,_ = pl_source.shape
        pl_source = pl_source.flatten(start_dim=2)
        sam_source = sam_source.flatten(start_dim=2)
        # reshaping to (batch,64*64,embed_dim)
        pl_source = pl_source.transpose(1,2).contiguous()
        sam_source = sam_source.transpose(1,2).contiguous()
        
        sam_source = self.sam_norm(sam_source)
        pl_source = self.pl_norm(pl_source)

        query:torch.Tensor = self.W_query(pl_source)
        key:torch.Tensor = self.W_key(sam_source)
        value:torch.Tensor = self.W_value(sam_source)

        attention = self.compute_att(query,key,value)

        attention = self.ff_layer_norm(pl_source+attention)
        out = self.ff(attention)
        out = self.ff_dropout(out+attention)

        out = out.transpose(1,2).view(batch,emed_dim,64,64).contiguous()

        return out
