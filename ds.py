import albumentations as A
import albumentations.augmentations.crops.functional as F
import cv2
import lmdb
import numpy as np
import os.path as op
import pickle
import six
import tempfile
import torch
import torchvision.transforms as T
from PIL import Image
from albumentations import CropNonEmptyMaskIfExists
from albumentations.pytorch import ToTensorV2
from copy import deepcopy
from jpeg2dct.numpy import load
from random import randint
from random import random
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import cfg
from get_qt import get_luma_qt_8x8

def load_qt(qt_path):
    with open(qt_path, 'rb') as fpk:
        pks_ = pickle.load(fpk)
    pks = {}
    for k, v in pks_.items():
        pks[k] = torch.LongTensor(v)
    return pks


def load_data(idx, lmdb):
    img_key = 'image-%09d' % idx
    img_buf = lmdb.get(img_key.encode('utf-8'))
    buf = six.BytesIO()
    buf.write(img_buf)
    buf.seek(0)
    img = Image.open(buf)
    lbl_key = 'label-%09d' % idx
    lbl_buf = lmdb.get(lbl_key.encode('utf-8'))
    mask = (cv2.imdecode(np.frombuffer(lbl_buf, dtype=np.uint8), 0) != 0).astype(np.uint8)
    return img, mask


def load_jpeg_record(record_path):
    with open(record_path, 'rb') as f:
        record = pickle.load(f)
    return record


def bbox_2_mask(bbox, ori_h, ori_w, expand_ratio=0.1):
    ocr_mask = np.zeros([ori_h, ori_w])
    for char_bbox in bbox:
        x1, y1, x2, y2 = char_bbox
        w = x2 - x1
        h = y2 - y1
        x1 = int(max(0, x1 - w * expand_ratio))
        y1 = int(max(0, y1 - h * expand_ratio))
        x2 = int(min(ori_w, x2 + w * expand_ratio))
        y2 = int(min(ori_h, y2 + h * expand_ratio))
        ocr_mask[int(y1):int(y2), int(x1):int(x2)] = 1
    return ocr_mask


def multi_jpeg(img, num_jpeg, min_qf, upper_bound, jpeg_record=None):
    with tempfile.NamedTemporaryFile(delete=True, suffix='.jpg') as tmp:
        img = img.convert("L")
        im_ori = img.copy()
        qf_record = []
        if jpeg_record is not None:
            num_jpeg = len(jpeg_record)
        for each_jpeg in range(num_jpeg):
            if jpeg_record is not None:
                qf = jpeg_record[each_jpeg]
            else:
                qf = randint(min_qf, upper_bound)
            qf_record.append(qf)
            img.save(tmp.name, "JPEG", quality=int(qf))
            img.close()
            img = Image.open(tmp.name)

        img = Image.open(tmp.name)
        img = img.convert('RGB')
        try:
            dct_y, _, _ = load(tmp.name, normalized=False)
        except:
            with tempfile.NamedTemporaryFile(delete=True) as tmp1:
                qf = 100
                qf_record = [100]
                im_ori.save_ckpt(tmp1, "JPEG", quality=qf)
                img = Image.open(tmp1)
                img = img.convert('RGB')
                dct_y, _, _ = load(tmp1.name, normalized=False)

    # dct_y [h, w, nb]
    rows, cols, _ = dct_y.shape
    dct = np.empty(shape=(8 * rows, 8 * cols))
    for j in range(rows):
        for i in range(cols):
            dct[8 * j: 8 * (j + 1), 8 * i: 8 * (i + 1)] = dct_y[j, i].reshape(8, 8)
    # dct to int32
    dct = np.int32(dct)
    return dct, img, qf_record


class AlignCrop(CropNonEmptyMaskIfExists):
    def apply(self, img, crop_coords, **params):
        x_min, y_min, x_max, y_max = crop_coords
        x_diff = x_min % 8
        x_min, x_max = x_min - x_diff, x_max - x_diff
        y_diff = y_min % 8
        y_min, y_max = y_min - y_diff, y_max - y_diff
        return F.crop(img, x_min, y_min, x_max, y_max)


class NonAlignCrop(CropNonEmptyMaskIfExists):
    def apply(self, img, crop_coords, **params):
        x_min, y_min, x_max, y_max = crop_coords
        h, w = img.shape[:2]
        x_diff = x_min % 8
        y_diff = y_min % 8

        if x_diff == 0 and y_diff == 0:  # if aligned, make it non-aligned
            # Try to shift the entire crop window by 1 pixel
            # Strategy: prefer shifting right/down if possible, otherwise left/up

            # For x-direction
            if x_max < w:  # Can shift right
                x_min += 1
                x_max += 1
            elif x_min > 0:  # Can shift left
                x_min -= 1
                x_max -= 1

            # For y-direction
            if y_max < h:  # Can shift down
                y_min += 1
                y_max += 1
            elif y_min > 0:  # Can shift up
                y_min -= 1
                y_max -= 1

        return F.crop(img, x_min, y_min, x_max, y_max)


def get_align_aug():
    return A.Compose([
        AlignCrop(cfg.img_size, cfg.img_size, p=1),
        A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
        A.OneOf([
            A.Downscale(scale_range=(0.5, 0.99), p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.RandomToneCurve(p=1),
                A.Sharpen(p=1),
            ], p=1),
        ], p=0.5)
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=16,
                                     min_visibility=0.2,
                                     label_fields=[]))


def get_non_align_aug():
    return A.Compose([
        A.RandomScale(scale_limit=(-0.5, 0.5), p=0.5),
        NonAlignCrop(cfg.img_size, cfg.img_size, p=1),
        A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
        ], p=1),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 9), sigma_limit=(0.5, 0.9), p=0.5),
            A.OneOf([
                A.GaussNoise(p=1),
                A.ISONoise(p=1),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
                A.RandomToneCurve(p=1),
                A.Sharpen(p=1),
            ], p=0.5),
        ], p=0.5)
    ], p=1, bbox_params=A.BboxParams(format='pascal_voc',
                                     min_area=16,
                                     min_visibility=0.2,
                                     label_fields=[]))


img_totsr = T.Compose([T.ToTensor(),
                       T.Normalize(mean=(0.485, 0.455, 0.406),
                                   std=(0.229, 0.224, 0.225))])

mask_totsr = ToTensorV2()


class TrainDs(Dataset):
    def __init__(self):
        lmdb_path = op.join(cfg.data_root, 'DocTamperV1-TrainingSet')
        self.lmdb = lmdb.open(lmdb_path, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb.begin(write=False) as txn:
            self.sample_n = int(txn.get('num-samples'.encode('utf-8')))
        self.ocr_dir = op.join(cfg.ocr_root, 'TrainingSet/char_seg')
        self.qts = load_qt(cfg.qt_path)

        self.S = cfg.init_S
        self.T = cfg.step_per_epoch
        self.min_qf = cfg.min_qf
        self.ds_len = cfg.ds_len

        self.align_aug = get_align_aug()
        self.non_align_aug = get_non_align_aug()
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

    def __len__(self):
        return self.ds_len

    def __getitem__(self, _):
        with self.lmdb.begin(write=False) as lmdb:
            index = randint(0, self.sample_n - 1)
            img_name = '%06d' % index
            img, mask = load_data(index, lmdb)

        # load char seg
        char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
        if op.exists(char_seg_path):
            with open(char_seg_path, 'rb') as f:
                c_bbox = pickle.load(f)
        else:
            c_bbox = []

        img = np.array(img)

        if random() > 0.5:  # DCT grid align sample
            aug_func = self.align_aug
            is_align = True
        else:  # non-align sample
            aug_func = self.non_align_aug
            is_align = False

        aug_out = aug_func(image=img, mask=mask, bboxes=c_bbox)
        img, mask, c_bbox = aug_out['image'], aug_out['mask'], aug_out['bboxes']
        h, w = mask.shape
        ocr_mask = bbox_2_mask(c_bbox, h, w)
        img = Image.fromarray(img)

        min_qf = max(int(round(100 - (self.S / self.T))), 75)
        num_jpeg = randint(1, 3)

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=num_jpeg,
                                   min_qf=min_qf,
                                   upper_bound=100)

        qf = qfs[-1]
        qt = self.qts[qf]
        img = self.img_totsr(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt,
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'min_qf': min_qf,
            'is_align': is_align
        }


class DtdValDs(Dataset):
    def __init__(self, val_name, is_sample=False):
        lmdb_path = op.join(cfg.data_root, f'DocTamperV1-{val_name}')
        self.lmdb = lmdb.open(lmdb_path, max_readers=64, readonly=True, lock=False, readahead=False, meminit=False)
        with self.lmdb.begin(write=False) as txn:
            self.sample_n = int(txn.get('num-samples'.encode('utf-8')))
        if is_sample:
            self.sample_n = cfg.val_sample_n

        self.qts = load_qt(cfg.qt_path)
        self.ocr_dir = op.join(cfg.ocr_root, f'{val_name}/char_seg')
        self.jpeg_record = load_jpeg_record(op.join(cfg.jpeg_record_dir, f'DocTamperV1-{val_name}_{cfg.min_qf}.pk'))
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        with self.lmdb.begin(write=False) as lmdb:
            img_name = '%06d' % index
            img, mask = load_data(index, lmdb)
            h, w = mask.shape

        char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
        if op.exists(char_seg_path):
            with open(char_seg_path, 'rb') as f:
                c_bbox = pickle.load(f)
        else:
            c_bbox = []

        # augment
        if cfg.val_aug is not None:
            img = np.array(img)
            aug = cfg.val_aug(image=img, mask=mask, bboxes=c_bbox)
            img, mask, c_bbox = aug['image'], aug['mask'], aug['bboxes']
            h, w = mask.shape
            ocr_mask = bbox_2_mask(c_bbox, h, w)
            img = Image.fromarray(img)
        else:
            ocr_mask = bbox_2_mask(c_bbox, h, w)

        if cfg.shift_1p:
            img = np.array(img)
            img = np.roll(img, 1, axis=0)
            img = np.roll(img, 1, axis=1)
            img = Image.fromarray(img)
            mask = np.roll(mask, 1, axis=0)
            mask = np.roll(mask, 1, axis=1)
            ocr_mask = np.roll(ocr_mask, 1, axis=0)
            ocr_mask = np.roll(ocr_mask, 1, axis=1)

        if cfg.multi_jpeg_val:
            record = list(self.jpeg_record[index][-2:])
        else:
            if cfg.jpeg_record:
                record = cfg.jpeg_record
            else:
                record = [100]

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=-1,
                                   min_qf=-1,
                                   upper_bound=-1,
                                   jpeg_record=record)

        qt = self.qts[100]  # self.qts[qfs[-1]]
        img = self.img_totsr(img)
        ori_img = np.array(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt,
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'ori_img': ori_img,
        }


class GeneralValDs(Dataset):
    def __init__(self, ds_name, is_sample=False):
        pkl_path = op.join(cfg.pkl_dir, f'{ds_name}.pkl')
        with open(pkl_path, 'rb') as f:
            self.path_list = pickle.load(f)

        self.sample_n = len(self.path_list)
        if is_sample:
            self.sample_n = cfg.val_sample_n

        self.qts = load_qt(cfg.qt_path)
        self.mask_totsr = mask_totsr
        self.img_totsr = img_totsr

        self.resize_func = A.Compose(
            [
                A.LongestMaxSize(cfg.val_max_size, p=1.0),
                # Add other transforms here if needed, e.g., A.HorizontalFlip(p=0.5)
            ],
            additional_targets={'mask2': 'mask'}  # Specify the second mask as type 'mask'
        )

    def __len__(self):
        return self.sample_n

    def __getitem__(self, index):
        img_path, mask_path, ocr_path = self.path_list[index]


        img_name = op.basename(img_path).split('.')[0]
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        mask = (cv2.imread(mask_path, 0) != 0).astype(np.uint8)
        ocr_mask = (cv2.imread(ocr_path, 0) != 0).astype(np.uint8)

        # char_seg_path = op.join(self.ocr_dir, img_name + '.pkl')
        # with open(char_seg_path, 'rb') as f:
        #     c_bbox = pickle.load(f)

        if h > cfg.val_max_size or w > cfg.val_max_size:
            img = np.array(img)
            aug = self.resize_func(image=img, mask=mask, mask2=ocr_mask)
            img, mask, ocr_mask = aug['image'], aug['mask'], aug['mask2']

        img = Image.fromarray(img)

        dct, img, qfs = multi_jpeg(deepcopy(img),
                                   num_jpeg=-1,
                                   min_qf=-1,
                                   upper_bound=-1,
                                   jpeg_record=[100])
        qt = self.qts[qfs[-1]]
        img = self.img_totsr(img)
        ori_img = np.array(img)
        mask = self.mask_totsr(image=mask.copy())['image']
        ocr_mask = self.mask_totsr(image=ocr_mask.copy())['image']

        return {
            'img': img,
            'dct': np.clip(np.abs(dct), 0, 20),
            'qt': qt.clamp(0, 63),
            'mask': mask.long(),
            'ocr_mask': ocr_mask.long(),
            'img_name': img_name,
            'ori_img': ori_img,
        }


def get_train_dl(world_size, rank, dp=False):
    ds = TrainDs()
    sampler = DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=True) if not dp else None
    dl = DataLoader(dataset=ds, batch_size=cfg.train_bs, num_workers=cfg.dl_workers, sampler=sampler)
    return dl


def pad_collate(batch, pad_value=0.0, mask_ignore_index=-1):
    """
    Args
    ----
    batch : list of (image, mask) where
            image:  C x H x W  float tensor
            mask :  H x W      long tensor  (class indices)  or  None
    pad_value : value to fill padded image pixels
    mask_ignore_index : index to fill padded mask pixels

    Returns
    -------
    images      : B x C x H_max x W_max   float tensor
    masks       : B x   H_max x W_max     long  tensor  (same padding)
    orig_sizes  : B x 2 (H, W)            long  tensor  (original shapes)
    """
    imgs = [item['img'] for item in batch]
    masks = [item['mask'] for item in batch]
    ocr_masks = [item['ocr_mask'] for item in batch]
    img_names = [item['img_name'] for item in batch]
    dcts = [item['dct'] for item in batch]
    qts = [item['qt'] for item in batch]

    sizes = torch.tensor([[im.shape[-2], im.shape[-1]] for im in imgs],
                         dtype=torch.long)  # B x 2

    H_max = int(sizes[:, 0].max())
    W_max = int(sizes[:, 1].max())

    # H_max should be divisible by 8
    divide_by = 16
    if H_max % divide_by != 0:
        H_max = (H_max // divide_by + 1) * divide_by
    if W_max % divide_by != 0:
        W_max = (W_max // divide_by + 1) * divide_by
    H_max = W_max = max(H_max, W_max)  # make it square
    padded_imgs = []
    padded_masks = []
    padded_ocr_masks = []
    padded_dcts = []

    for im, m, ocr_m, dct, qt in zip(imgs, masks, ocr_masks, dcts, qts):
        C, H, W = im.shape
        pad_h = H_max - H
        pad_w = W_max - W
        # pad order for F.pad is (left, right, top, bottom)
        im_p = torch.nn.functional.pad(im, (0, pad_w, 0, pad_h), value=pad_value)
        padded_imgs.append(im_p)

        # if m is None:
        #     # create dummy mask filled with ignore_index
        #     m_p = torch.full((H_max, W_max),
        #                      fill_value=mask_ignore_index,
        #                      dtype=torch.long,
        #                      device=im.device)
        # else:
        C, H, W = m.shape
        pad_h = H_max - H
        pad_w = W_max - W
        m_p = torch.nn.functional.pad(m, (0, pad_w, 0, pad_h), value=mask_ignore_index)
        padded_masks.append(m_p)

        C, H, W = ocr_m.shape
        pad_h = H_max - H
        pad_w = W_max - W
        ocr_m_p = torch.nn.functional.pad(ocr_m, (0, pad_w, 0, pad_h), value=mask_ignore_index)
        padded_ocr_masks.append(ocr_m_p)
        H, W = dct.shape
        pad_h = H_max - H
        pad_w = W_max - W
        # pad dct in np array
        dct_p = torch.nn.functional.pad(torch.tensor(dct), (0, pad_w, 0, pad_h), value=pad_value)
        padded_dcts.append(dct_p)

    b = 1

    # pad on batch dim, b should be equal to cfg.val_bs
    if len(padded_imgs) < b:
        b_diff = b - len(padded_imgs)
        for _ in range(b_diff):
            padded_imgs.append(torch.full((3, H_max, W_max), fill_value=pad_value))
            padded_masks.append(torch.full((1, H_max, W_max), fill_value=0, dtype=torch.long))
            padded_ocr_masks.append(torch.full((1, H_max, W_max), fill_value=0, dtype=torch.long))
            padded_dcts.append(torch.full((H_max, W_max), fill_value=0, dtype=torch.long))
            qts.append(torch.full((8, 8), fill_value=1, dtype=torch.long))
            img_names.append('padding')
            sizes = torch.cat([sizes, torch.tensor([[H_max, W_max]], dtype=torch.long)], dim=0)

    batch_imgs = torch.stack(padded_imgs)  # B x C x H_max x W_max
    batch_masks = torch.stack(padded_masks)  # B x H_max x W_max
    batch_ocr_masks = torch.stack(padded_ocr_masks)  # B x H_max x W_max
    batch_dcts = torch.stack(padded_dcts)  # B x C x H_max x W_max
    batch_qts = torch.stack(qts)  # B x C x H_max x W_max

    return batch_imgs, batch_dcts, batch_qts, batch_masks, batch_ocr_masks, list(
        img_names), sizes  # sizes keeps originals


def get_val_dl(world_size, rank, dp=False):
    dl_list = {}
    for val_name in cfg.val_name_list:

        is_sample = False
        if 'sample' in val_name:
            val_name = val_name.replace('_sample', '')
            is_sample = True

        if val_name in ['FCD', 'SCD', 'TestingSet']:
            ds = DtdValDs(val_name, is_sample)
            b = cfg.val_bs
        else:
            ds = GeneralValDs(val_name, is_sample)
            b = cfg.val_bs

        sampler = DistributedSampler(dataset=ds, num_replicas=world_size, rank=rank, shuffle=False) if not dp else None
        dl = DataLoader(dataset=ds, batch_size=b, num_workers=cfg.dl_workers, sampler=sampler,
                        collate_fn=pad_collate)
        dl_list[val_name] = dl

    return dl_list


if __name__ == '__main__':

    ds = TrainDs()
    from tqdm import tqdm

    for i in tqdm(range(50000)):
        tmp = ds.__getitem__(i)
        i = 0
    ds = DtdValDs(roots='/data/jesonwong47/DocTamper/DocTamperV1/DocTamperV1-FCD',
                  minq=75)
    ds.__getitem__(0)
