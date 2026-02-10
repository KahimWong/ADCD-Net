# <editor-fold desc="header">
import os

import cfg

os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
import torch

torch.set_num_threads(1)

import os.path as op
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import datetime
import logging
import cv2
import PIL.Image as Image
from copy import deepcopy

from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.amp import autocast, GradScaler

from model.model import ADCDNet
from loss.soft_ce_loss import SoftCrossEntropyLoss
from loss.lovasz_loss import LovaszLoss
from utils import AverageMeter
from ds import get_train_dl, get_val_dl, multi_jpeg, load_qt

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(asctime)s] %(message)s', datefmt='%m-%d %H:%M:%S')


# </editor-fold>


class Trainer:
    def __init__(self, rank, world_size):
        super(Trainer, self).__init__()
        self.rank = rank
        self.world_size = world_size
        if self.rank == 0:
            now_time = datetime.datetime.now()
            now_time = 'Log_v%02d%02d%02d%02d/' % (now_time.month, now_time.day, now_time.hour, now_time.minute)
            exp_dir = op.join(cfg.root, f'exp_out/{cfg.exp_root_name}', now_time)
            tb_log = op.join(exp_dir, 'tb_log')
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(tb_log, exist_ok=True)
            self.tb_writer = SummaryWriter(tb_log)
            self.ckpt_dir = op.join(exp_dir, 'ckpt')
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # data loader
        self.train_dl = get_train_dl(self.world_size, self.rank)
        self.val_dls = get_val_dl(self.world_size, self.rank)
        # model
        self.model = ADCDNet().to(f'cuda:{self.rank}')
        self.load_ckpt(cfg.ckpt)
        self.model = DDP(self.model, device_ids=[self.rank], find_unused_parameters=True)
        # optimizer and scheduler
        self.optimizer = AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, len(self.train_dl) * cfg.epochs, eta_min=cfg.min_lr)
        # loss
        self.loc_ce = SoftCrossEntropyLoss(smooth_factor=0.1, reduction="mean", ignore_index=None)
        self.loc_lovasz = LovaszLoss(mode='multiclass', per_image=True)
        self.rec_l1 = nn.L1Loss()
        self.align_ce = nn.CrossEntropyLoss()

        self.scaler = GradScaler()
        self.eps = 1e-8

    def train(self):
        step = 1
        self.model.train()

        for epoch in range(1, cfg.epochs + 1):
            losses_record = defaultdict(AverageMeter)

            if epoch != 1:
                self.train_dl.dataset.S += cfg.step_per_epoch

            self.train_dl.sampler.set_epoch(epoch)
            tqdm_fn = tqdm if self.rank == 0 else lambda x: x

            for items in tqdm_fn(self.train_dl):

                # forward
                img, dct, qt, mask, ocr_mask, is_align, min_qf = \
                    (
                        items['img'].to(f'cuda:{self.rank}'),
                        items['dct'].to(f'cuda:{self.rank}'),
                        items['qt'].to(f'cuda:{self.rank}'),
                        items['mask'].to(f'cuda:{self.rank}'),
                        items['ocr_mask'].to(f'cuda:{self.rank}'),
                        items['is_align'].to(f'cuda:{self.rank}'),
                        items['min_qf'][0]
                    )

                with autocast(device_type='cuda', dtype=torch.float16):
                    logits, norm_feats, align_logits, rec_items, focal_losses = self.model(
                        img, dct, qt, mask, ocr_mask, is_train=True)

                # loss

                # reconstruction loss
                rec, norm_dct = rec_items
                img_l1_loss = self.rec_l1(rec[:, :3], img)
                dct_l1_loss = self.rec_l1(rec[:, -1], norm_dct)
                rec_loss = cfg.rec_w * (img_l1_loss + dct_l1_loss)

                # feature norm loss
                norm_losses = []
                for feat in norm_feats:
                    norm_losses.append(feat.norm(dim=1).mean())
                norm_loss = cfg.norm_w * torch.stack(norm_losses).mean()

                # dct align score loss
                align_loss = self.align_ce(align_logits, is_align.long())

                # focal loss
                focal_loss = [cfg.focal_w * (loss.sum() / (loss != 0).sum()) for loss in focal_losses]
                focal_loss = torch.stack(focal_loss).sum()

                # localization loss
                ce_loss = cfg.ce_w * self.loc_ce(logits.float(), mask)
                iou_loss = self.loc_lovasz(logits.float(), mask)

                total_loss = ce_loss + iou_loss + rec_loss + align_loss + focal_loss + norm_loss

                with torch.no_grad():
                    f1, p, r = self.compute_f1(logits, mask)
                    align_acc = (align_logits.argmax(1) == is_align).float().mean().item()

                losses = {
                    'total': total_loss.item(),
                    'ce': ce_loss.item(),
                    'iou': iou_loss.item(),
                    'rec': rec_loss.item(),
                    'align_ce': align_loss.item(),
                    'focal': focal_loss.item(),
                    'norm': norm_loss.item(),
                    'f1': f1,
                    'align_acc': align_acc,
                    'min_qf': min_qf,
                }

                # backward
                self.scaler.scale(total_loss / cfg.accum_step).backward()

                if (step + 1) % cfg.accum_step == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                for name, loss in losses.items():
                    val_tensor = torch.tensor(loss).to(f'cuda:{self.rank}')
                    dist.reduce(val_tensor, dst=0, op=dist.ReduceOp.SUM)
                    if self.rank == 0:
                        avg_val = val_tensor.item() / self.world_size
                        losses_record[name].update(avg_val)

                if self.rank == 0:
                    if step % cfg.print_log_step == 0:
                        self.print_log(step, losses_record)
                        self.write_log(step, losses_record)

                if cfg.check_val or step % cfg.val_step == 0:
                    val_score = self.val()
                    if self.rank == 0:
                        self.save_ckpt(step, val_score)
                step += 1

                self.scheduler.step()

    def val(self):
        self.model.eval()
        with torch.no_grad():
            ds_f1_list = []
            for val_name, dl in self.val_dls.items():

                if self.rank == 0:
                    logging.info('Val Set: %s' % val_name)

                sum_f1 = torch.zeros(1, device=f'cuda:{self.rank}')
                sum_p = torch.zeros(1, device=f'cuda:{self.rank}')
                sum_r = torch.zeros(1, device=f'cuda:{self.rank}')
                num_images = torch.zeros(1, device=f'cuda:{self.rank}')

                for items in tqdm(dl, disable=(self.rank != 0)):
                    img, dct, qt, mask, ocr_mask, img_names, sizes = \
                        (
                            items[0].cuda(),
                            items[1].cuda(),
                            items[2].cuda(),
                            items[3].cuda(),
                            items[4].cuda(),
                            items[5],
                            items[6]
                        )

                    with autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(img, dct, qt, mask, ocr_mask, is_train=False)[0]

                    for logit, each_y, (h, w), name in zip(logits, mask, sizes, img_names):
                        if name != 'padding':
                            crop_logit = logit[..., :h, :w].unsqueeze(0)
                            crop_y = each_y[..., :h, :w].unsqueeze(0)
                            per_f1, per_p, per_r = self.compute_f1(crop_logit, crop_y)
                            sum_f1 += per_f1
                            sum_p += per_p
                            sum_r += per_r
                            num_images += 1.

                dist.reduce(sum_p, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(sum_r, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(sum_f1, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(num_images, dst=0, op=dist.ReduceOp.SUM)
                if self.rank == 0:
                    # p = sum_p.item() / num_images.item()
                    # r = sum_r.item() / num_images.item()
                    # f1 = 2 * p * r / (p + r + self.eps)
                    # logging.info('P:%.4f R:%.4f F1:%.4f' % (p, r, f1))
                    f1 = sum_f1.item() / num_images.item()
                    logging.info('AVG F1: %.4f' % f1)
                    ds_f1_list.append(f1)

            if self.rank == 0:
                total_f1 = np.mean(ds_f1_list)
            else:
                total_f1 = 0.0
            total_f1_tensor = torch.tensor(total_f1, device=f'cuda:{self.rank}')
            dist.broadcast(total_f1_tensor, src=0)
            total_f1 = total_f1_tensor.item()

        self.model.train()

        if self.rank == 0:
            logging.info('Score: %5.4f' % total_f1)

        return total_f1

    @torch.no_grad()
    def compute_f1(self, logit, y):
        pred = logit.argmax(1)  # ori [b,h,w]
        y_ = y.squeeze(1)
        matched = (pred * y_).sum((1, 2))
        pred_sum = pred.sum((1, 2))
        y_sum = y_.sum((1, 2))
        p = (matched / (pred_sum + self.eps)).mean().item()
        r = (matched / (y_sum + self.eps)).mean().item()
        f1 = (2 * p * r / (p + r + self.eps))
        return f1, p, r

    def write_log(self, cnt, losses_record):
        if self.rank == 0:
            for loss_name, loss_value in losses_record.items():
                self.tb_writer.add_scalar('losses/{}'.format(loss_name.strip()), loss_value.val, global_step=cnt)

    def print_log(self, step, losses_record):
        if self.rank != 0:
            return
        lr = self.optimizer.param_groups[0]['lr']
        output = 'Step: %6d; lr:%.2e;' % (step, lr)
        for name, loss in losses_record.items():
            output += ' %s: %5.4f;' % (name, loss.val)
        logging.info(output)

    def load_ckpt(self, ckpt_path):
        if ckpt_path is not None:
            self.ckpt = torch.load(cfg.ckpt, map_location='cpu', weights_only=True)
            miss, unexpect = self.model.load_state_dict(self.modify_cp_dict(self.ckpt['model']), strict=False)
            logging.info(f"Loaded model from {cfg.ckpt}. Missed keys: {miss}, Unexpected keys: {unexpect}")

    def modify_cp_dict(self, cp_dict):
        new_cp_dict = {}
        for key in cp_dict:
            new_key = key.replace('module.', '')
            new_cp_dict[new_key] = cp_dict[key]
        return new_cp_dict

    def save_ckpt(self, step, score):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': step,
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state_dict, op.join(self.ckpt_dir, 'Step%s_Score%5.4f.pth' % (step, score)))


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Choose any free port; 29500 is a common default
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    trainer = Trainer(rank, world_size)
    if cfg.mode == 'train':
        trainer.train()
    elif cfg.mode == 'val':
        trainer.val()


if __name__ == '__main__':
    world_size_ = torch.cuda.device_count()
    mp.spawn(main, args=(world_size_,), nprocs=world_size_, join=True)
