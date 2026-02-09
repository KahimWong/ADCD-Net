import os
import os.path as op
from glob import glob
from tqdm import tqdm


# config

def main():

    root = 'path/to/cutted_datasets_fakes'

    ds_name_list = ['Tampered-IC13_test', 'OSTF_test', 'T-SROIE_test', 'RealTextManipulation_test']

    pkl_dir = op.join(root, 'path_pkl')
    os.makedirs(pkl_dir, exist_ok=True)

    for ds_name in  ds_name_list:
        ds_dir = os.path.join(root, ds_name)
        ocr_dir = os.path.join(ds_dir, 'ocr')
        img_dir = os.path.join(ds_dir, 'images')
        mask_dir = os.path.join(ds_dir, 'masks')
        path_list = []
        img_list = glob(os.path.join(img_dir, '*'))
        for img_path in tqdm(img_list):
            img_name = os.path.basename(img_path)
            ocr_path = os.path.join(ocr_dir, img_name)
            mask_name = img_name.replace('.jpg', '.png')
            mask_path = os.path.join(mask_dir, mask_name)
            # check exist
            if not os.path.exists(ocr_path):
                print(f'OCR path does not exist: {ocr_path}')
                continue
            if not os.path.exists(mask_path):
                print(f'Mask path does not exist: {mask_path}')
                continue
            if not os.path.exists(img_path):
                print(f'Image path does not exist: {img_path}')
                continue
            path_list.append((img_path, mask_path, ocr_path))
        # save pkl
        save_path = os.path.join(pkl_dir, ds_name + '.pkl')
        with open(save_path, 'wb') as f:
            import pickle
            pickle.dump(path_list, f)

    return


if __name__ == '__main__':
    main()
