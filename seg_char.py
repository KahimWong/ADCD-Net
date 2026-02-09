import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
from paddleocr import TextDetection
from glob import glob
from tqdm import tqdm

class TextDetector:
    def __init__(self):
        self.model = TextDetection(model_name="PP-OCRv5_server_det")

    def get_mask(self, img_path, save_path):
        output = self.model.predict(input=img_path, batch_size=1)

        # Extract detection results (assuming single image input)
        res = output[0]
        polys = res['dt_polys']
        scores = res['dt_scores']  # Optional: can filter based on scores if needed

        # Load image to get dimensions
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Create binary mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for i, poly in enumerate(polys):
            if scores[i] > 0.5:  # Optional threshold for confidence
                cv2.fillPoly(mask, [poly.astype(np.int32)], 1)

        # Save the mask as PNG (multiply by 255 for visibility: white text regions on black background)
        cv2.imwrite(save_path, mask * 255)


if __name__ == '__main__':
    detector = TextDetector()

    root = 'path/to/cutted_datasets_fakes'

    ds_name_list = ['Tampered-IC13_test', 'OSTF_test', 'T-SROIE_test', 'RealTextManipulation_test']

    for ds_name in  ds_name_list:
        print(f'Processing dataset: {ds_name}')
        ds_dir = os.path.join(root, ds_name)
        ocr_dir = os.path.join(ds_dir, 'ocr')
        img_dir = os.path.join(ds_dir, 'images')
        os.makedirs(ocr_dir, exist_ok=True)
        img_list = glob(os.path.join(img_dir, '*'))
        for img_path in tqdm(img_list):
            img = cv2.imread(img_path)
            img_name = os.path.basename(img_path)
            ocr_path = os.path.join(ocr_dir, img_name)
            detector.get_mask(img_path=img_path, save_path=ocr_path)
