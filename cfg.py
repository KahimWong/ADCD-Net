import os.path as op

gpus = '0,1,2,3'
device_n = len(gpus.split(','))

mode = 'val'  # 'train', 'val'
check_val = False
train_bs = 5 # batch size per device
val_bs = 8
step_per_epoch = 1000
ds_len = sample_per_epoch = step_per_epoch * train_bs * device_n
print_log_step = 100
val_step = step_per_epoch * 10
epochs = 200
accum_step = 2  # Gradient accumulation steps

# ======= Set evaluation distortion here =======
multi_jpeg_val = True  # able to use multi jpeg distortion
jpeg_record = False  # manually set multi jpeg distortion record
min_qf = 75  # minimum jpeg quality factor
shift_1p = False  # shift 1 pixel for evaluation
init_S = 0
val_aug = None
# ======= Set evaluation distortion here =======

# ------------------ MODEL CFG -------------------

root = 'path/to/DocTamper'
ckpt = 'path/to/ADCDNet.pth'
docres_ckpt_path = 'path/to/docres.pkl'
pkl_dir = 'path/to/path_pkl'  # specify the directory where pkl files are stored

# ['TestingSet', 'FCD', 'SCD', 'T-SROIE_test', 'Tampered-IC13_test', 'RealTextManipulation_test', 'OSTF_test']
all_ds_name = ['TestingSet', 'FCD', 'SCD', 'T-SROIE_test', 'Tampered-IC13_test', 'RealTextManipulation_test', 'OSTF_test']
all_ds_name_s = [name + '_sample' for name in all_ds_name]  # append '_sample' to each name
val_name_list = all_ds_name_s if mode == 'train' else all_ds_name
val_sample_n = 100
val_max_size = 512

# -------------------- FIX ----------------------

data_root = op.join(root, 'DocTamperV1')
ocr_root = op.join(root, 'DocTamperData')
qt_path = op.join(root, 'exp_data/qt_table.pk')
jpeg_record_dir = op.join(root, 'exp_data/pks')
exp_root_name = 'ADCDNet'
lr = 3e-4
min_lr = 1e-5
weight_decay = 1e-4
img_size = 256
ce_w = 3
rec_w = 1
focal_w = 0.2
norm_w = 0.1
dl_workers = 0
total_step = step_per_epoch * epochs

