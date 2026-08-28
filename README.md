
# ADCD-Net

[![arXiv](https://img.shields.io/badge/arXiv-2507.16397-b31b1b.svg)](https://arxiv.org/abs/2507.16397)
[![Venue](https://img.shields.io/badge/Venue-ICCV-blue)](https://openaccess.thecvf.com/content/ICCV2025/html/Wong_ADCD-Net_Robust_Document_Image_Forgery_Localization_via_Adaptive_DCT_Feature_ICCV_2025_paper.html)
[![License](https://img.shields.io/github/license/KahimWong/ADCD-Net)](LICENSE)

> **ADCD-Net** addresses the challenging problem of document image forgery localization by leveraging adaptive DCT features alongside hierarchical content disentanglement to robustly detect tampered regions even under compression distortions.

![model_overview](./fig/model_overview.png)

---

## 📊 ForensicHub Benchmark (Doc Protocol)

![doc_protocol](./fig/docpro.png)

Evaluation follows the **Doc Protocol**: train on the DocTamper training set, evaluate on seven test sets. DocTamper FCD/SCD/Test sets are compressed once using the official DocTamper pickle QFs. Authentic images are skipped.

For more details, see [ForensicHub — Doc Protocol](https://github.com/scu-zjz/ForensicHub/issues/26).

---

## ⚙️ Environment Setup

| Dependency | Version |
|---|---|
| Python | 3.10.13 |
| PyTorch | 2.3.0+cu121 |
| albumentations | 2.0.8 |

---

## 📂 Data Preparation

### 1. Download DocTamper Data

Download the DocTamper dataset (Training, Testing, FCD, SCD) from:
👉 [DocTamper GitHub](https://github.com/qcf-568/DocTamper)

> `qt_table.pk` and `pks` (JPEG record pickle files) are available in the DocTamper repository.

### 2. Download ADCD-Net Checkpoints & OCR Masks

Download from Google Drive:
👉 [ADCD-Net Data (Google Drive)](https://drive.google.com/file/d/1-5BU3Bavs6SGJpaByua_FhDuUJGoo-iS/view?usp=sharing)

The archive contains:

```
ADCDNet.pth          # ADCD-Net model checkpoint
docres.pkl           # DocRes backbone checkpoint
DocTamperOCR/        # Pre-generated OCR mask directory
├── TrainingSet/     # Training set OCR masks
├── TestingSet/      # Testing set OCR masks
├── FCD/             # FCD dataset OCR masks
└── SCD/             # SCD dataset OCR masks
```

### 3. Download Doc Protocol Cross-Domain Test Sets

Download the 4 cross-domain test sets (T-SROIE, OSTF, TPIC-13, RTM) from:
👉 [Doc Protocol Data]([https://drive.google.com/drive/folders/1xn8mELN8etQwRo_PgS5XV6XTKCZasz_A?usp=drive_link](https://huggingface.co/datasets/Jason37437/Doc-Protocol-Data)) — `cutted_datasets_fakes.zip`

---

## 🔤 Get OCR Masks

OCR character segmentation masks are generated using `seg_char.py`, which requires **PaddleOCR**.

Install PaddlePaddle and PaddleOCR by following the official guide:
👉 [PaddleOCR Installation](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)

Then run:

```bash
python seg_char.py
```

---

## 🚀 Training

ADCD-Net is trained on **4 × NVIDIA GeForce RTX 4090 (24 GB)** with:
- **100k** training steps
- **Batch size:** 40 (10 per GPU × 4 GPUs with gradient accumulation)
- **Training time:** ~27 hours

**Steps:**

1. Configure paths in `cfg.py`:

```python
mode = 'train'
root = 'path/to/DocTamper'
docres_ckpt_path = 'path/to/docres.pkl'
```

2. Launch training:

```bash
python main.py
```

---

## 📈 Evaluation

Reproduce the ForensicHub Doc Protocol results with the following steps:

1. Generate OCR masks for the 4 cross-domain sets using `seg_char.py`.
2. Generate path pickle files for the 4 sets using `build_path_pkl.py`.
3. Configure `cfg.py` for evaluation:

```python
mode = 'val'
all_ds_name = ['TestingSet', 'FCD', 'SCD', 'T-SROIE_test', 'Tampered-IC13_test', 'RealTextManipulation_test', 'OSTF_test']
pkl_dir = 'path/to/path_pkl'
```

4. Run evaluation:

```bash
python main.py
```

---

## 📝 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{wong2025adcd,
  title={ADCD-Net: Robust Document Image Forgery Localization via Adaptive DCT Feature and Hierarchical Content Disentanglement},
  author={Wong, Kahim and Zhou, Jicheng and Wu, Haiwei and Si, Yain-Whar and Zhou, Jiantao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
}
```
