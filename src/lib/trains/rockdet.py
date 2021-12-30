from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss
from models.losses import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, \
    CrossEntropyRegSumLoss, CrossEntropyRegLoss
from models.decode import rockdet_decode
from models.utils import _sigmoid
from utils.debugger import Debugger
from utils.post_process import rockdet_post_process
from utils.oracle_utils import gen_oracle_map
from .base_trainer import BaseTrainer


class RockdetLoss(torch.nn.Module):
    def __init__(self, opt):
        super(RockdetLoss, self).__init__()
        self.opt = opt
        self.crit = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_wh = torch.nn.L1Loss(reduction='sum') if opt.dense_wh else \
                RegWeightedL1Loss() if opt.cat_spec_wh else self.crit_reg
        self.crit_ang = CrossEntropyRegSumLoss(self.opt.ang_anchors) \
            if opt.dense_wh else CrossEntropyRegLoss(self.opt.ang_anchors)

    def forward(self, outputs, batch):
        opt = self.opt
        hm_loss, wh_loss, ang_loss, ang_regr_loss, ang_clf_loss, off_loss = 0, 0, 0, 0, 0, 0
        for s in range(opt.num_stacks):
            output = outputs[s]
            
            if opt.ang_anchors:
                ang_regr = _sigmoid(output['ang'][:,
                    opt.ang_anchors:opt.ang_anchors + 1, :, :])
                output['ang'] = torch.cat([output['ang'][:, :-1, :, :],
                                           ang_regr * (1 / opt.ang_anchors)],
                                          axis=1)
            else:
                ang_regr = _sigmoid(output['ang'])
                
            if not opt.mse_loss:
                output['hm'] = _sigmoid(output['hm'])

            if opt.eval_oracle_hm:
                output['hm'] = batch['hm']
            if opt.eval_oracle_wh:
                output['wh'] = torch.from_numpy(gen_oracle_map(
                    batch['wh'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['wh'].shape[3], output['wh'].shape[2])).to(opt.device)
            if opt.eval_oracle_ang:
                output['ang'] = torch.from_numpy(gen_oracle_map(
                    batch['ang'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['ang'].shape[3], output['ang'].shape[2])).to(opt.device)
            if opt.eval_oracle_offset:
                output['reg'] = torch.from_numpy(gen_oracle_map(
                    batch['reg'].detach().cpu().numpy(),
                    batch['ind'].detach().cpu().numpy(),
                    output['reg'].shape[3], output['reg'].shape[2])).to(opt.device)

            hm_loss += self.crit(output['hm'], batch['hm']) / opt.num_stacks
            if opt.wh_weight > 0:
                if opt.dense_wh:
                    mask_weight = batch['dense_wh_mask'].sum() + 1e-4
                    wh_loss += (
                        self.crit_wh(output['wh'] * batch['dense_wh_mask'],
                                     batch['dense_wh'] * batch['dense_wh_mask']) /
                        mask_weight) / opt.num_stacks
                    ang_loss += (
                        self.crit_ang(output['ang'], batch['dense_ang'],
                                      batch['dense_ang_mask']) /
                        mask_weight) / opt.num_stacks
                elif opt.cat_spec_wh:
                    raise NotImplementedError
                    # wh_loss += self.crit_wh(
                    #     output['wh'], batch['cat_spec_mask'],
                    #     batch['ind'], batch['cat_spec_wh']) / opt.num_stacks
                    # ang_loss += self.crit_ang(
                    #     output['ang'], batch['cat_spec_mask'],
                    #     batch['ind'], batch['cat_spec_ang']) / opt.num_stacks
                else:
                    wh_loss += self.crit_wh(
                        output['wh'], batch['reg_mask'],
                        batch['ind'], batch['wh']) / opt.num_stacks
                    _ang_clf_loss, _ang_regr_loss = self.crit_ang(
                        output['ang'], batch['reg_mask'],
                        batch['ind'], batch['ang'])
                    ang_clf_loss += _ang_clf_loss / opt.num_stacks
                    ang_regr_loss += _ang_regr_loss / opt.num_stacks
                    ang_loss += (_ang_clf_loss + _ang_regr_loss) / opt.num_stacks

            if opt.reg_offset and opt.off_weight > 0:
                off_loss += self.crit_reg(output['reg'], batch['reg_mask'],
                                          batch['ind'], batch['reg']) / opt.num_stacks

        loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + \
               opt.off_weight * off_loss + opt.ang_weight * ang_loss
        loss_stats = {'loss': loss, 'hm_loss': hm_loss,
                      'wh_loss': wh_loss, 'ang_loss': ang_loss,
                      'ang_regr_loss': ang_regr_loss,
                      'ang_clf_loss': ang_clf_loss,
                      'off_loss': off_loss}
        return loss, loss_stats


class RockdetTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(RockdetTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'wh_loss', 'ang_loss', 'ang_regr_loss', 'ang_clf_loss', 'off_loss']
        loss = RockdetLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        opt = self.opt
        reg = output['reg'] if opt.reg_offset else None
        if opt.ang_anchors:
            ang_regr = output['ang'][:, -1, :, :][:, None, :, :]
            ang_clf = torch.argmax(output['ang'][:, :-1, :, :], dim=1, keepdim=True) * (1 / opt.ang_anchors)
            out_ang = ang_regr + ang_clf
            dets = rockdet_decode(
                output['hm'], output['wh'], out_ang, reg=reg,
                cat_spec_wh=opt.cat_spec_wh, K=opt.K, ang_anchors=opt.ang_anchors)
        else:
            dets = rockdet_decode(
                output['hm'], output['wh'], output['ang'], reg=reg,
                cat_spec_wh=opt.cat_spec_wh, K=opt.K, ang_anchors=opt.ang_anchors)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets[:, :, :4] *= opt.down_ratio
        dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
        dets_gt[:, :, :4] *= opt.down_ratio
        for i in range(1):
            debugger = Debugger(
                dataset=opt.dataset, ipynb=(opt.debug == 3), theme=opt.debugger_theme)
            img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            img = np.clip(((img * opt.std + opt.mean) * 255.), 0, 255).astype(np.uint8)
            pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hm')
            debugger.add_blend_img(img, gt, 'gt_hm')
            debugger.add_img(img, img_id='out_pred')
            for k in range(len(dets[i])):
                if dets[i, k, 4] > opt.center_thresh:
                    try:
                        debugger.add_coco_ellipse(dets[i, k, :5], dets[i, k, -1],
                                                  dets[i, k, 5], img_id='out_pred')
                    except:
                        pass

            debugger.add_img(img, img_id='out_gt')
            for k in range(len(dets_gt[i])):
                if dets_gt[i, k, 5] > opt.center_thresh:
                    debugger.add_coco_ellipse(dets_gt[i, k, :5], dets_gt[i, k, -1],
                                              dets_gt[i, k, 5], img_id='out_gt')

            if opt.debug == 4:
                debugger.save_all_imgs(opt.debug_dir, prefix='{}'.format(iter_id))
            else:
                debugger.show_all_imgs(pause=True)

    def save_result(self, output, batch, results):
        reg = output['reg'] if self.opt.reg_offset else None
        dets = rockdet_decode(
            output['hm'], output['wh'], output['ang'], reg=reg,
            cat_spec_wh=self.opt.cat_spec_wh, K=self.opt.K, use_sine_loss=opt.use_sine_loss)
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets_out = rockdet_post_process(
            dets.copy(), batch['meta']['c'].cpu().numpy(),
            batch['meta']['s'].cpu().numpy(),
            output['hm'].shape[2], output['hm'].shape[3], output['hm'].shape[1])
        results[batch['meta']['img_id'].cpu().numpy()[0]] = dets_out[0]

