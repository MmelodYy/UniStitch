<h1>
  UniStitch: Unifying Semantic and Geometric Features for Image Stitching
</h1>

<p align="center">
  <img src="./demo.gif" alt="Demo" width="100%" height="600">
</p>

> **[UniStitch: Unifying Semantic and Geometric Features for Image Stitching](https://arxiv.org/abs/2603.10568)**
>
>  [Yuan Mei](), [Lang Nie](https://nie-lang.github.io/),[Kang Liao](https://kangliao929.github.io/), [Yunqiu Xu](), [Chunyu Lin](), [Bin Xiao]()

>
> [![arXiv](https://img.shields.io/badge/arXiv-2603.10568-b31b1b.svg)](https://arxiv.org/abs/2603.10568)
> [![Project Page](https://img.shields.io/badge/Project%20Page-GitHub-blue)]()


## 📊 Dataset

We use the UDIS dataset to train and evaluate our method. Please refer to **[UDIS](https://github.com/nie-lang/UnsupervisedDeepImageStitching)** for more details about this dataset (**Note**: We noticed that the image `000001.jpg` is missing in the `testing/input1` folder of the original UDIS dataset. We have supplemented this missing image, which can be downloaded from this [link](https://drive.google.com/file/d/1M_-lqNdICT4tM3CWLK2WOcPCIzeQdc4d/view?usp=sharing)). Meanwhile, for cross-scenario validation, we used 147 pairs of classical image stitching datasets collected by **[RopStitch](https://github.com/MmelodYy/RopStitch/tree/main)**. You can download from this [link](https://drive.google.com/file/d/1_F7M7DN7K4BjZPEcez7XS6TUpE3iEX8f/view).

---

## 💻 Requirements

| Package | Version |
|---------|---------|
| numpy | >= 1.19.5 |
| pytorch | >= 1.7.1 |
| scikit-image | >= 0.15.0 |
| tensorboard | >= 2.9.0 |

---

## 🔧 Data Preparation

### Keypoint Extraction for New Datasets (Optional)
Before starting training, you need to extract keypoints for the input images. You can use any keypoint extractor of your choice. We provide two versions:

- **LightGLUE-based extractor** (`keypoint_tool/LightGLUE/get_keypoint_from_lightGlue.py`)
- **OpenCV-based extractor** (`keypoint_tool/get_keypoint_from_opencv.py`)

After extraction, place the generated keypoint files in the corresponding dataset folders.

> **Important**: If you modify the database name or key names during the keypoint extraction process, ensure that:
> - The saved LMDB database name matches `self.lmdb_path` in `/Codes/datasets.py`
> - The key names correspond to those used in the `_get_lmdb_data` function

### Quick Start (Pre-extracted Keypoints for UDIS and Classical Datasets)
To simplify the process, we provide pre-extracted keypoint files (generated with LightGLUE) for both the UDIS and Classical datasets. You can download them from this [link](https://drive.google.com/drive/folders/1kUFp0UMmOVkUQ9BRKFnCnOCVJQRtOIxX?usp=sharing). Then put them in the corresponding dataset folders.

```
 e.g., 
/testing/
├── input1/
├── input2/
└── superpoint_lmdb/
```

## ✈️ Training

### Step 1: Stage 1 Training

```bash
cd ./Codes/
python train_stage1.py
```

### Step 2: Stage 2 Training
```
python train_stage2.py
```
## 🖼️ Testing 
Our pretrained models can be available at [Google Drive](https://drive.google.com/file/d/1soyLMV4j5x6dfWEVOUMhpS9U4klPRvMA/view?usp=sharing).

```
python test.py
```

## 🎯 Fine-tuning

```
python test_finetune.py
```

## 📚 Citation

If you find UniStitch useful for your research or applications, please cite our paper using the following BibTeX:

```bibtex
  @inproceedings{Mei2026UniStitchUS,
  title={UniStitch: Unifying Semantic and Geometric Features for Image Stitching},
  author={Yuan Mei and Lang Nie and Kang Liao and Yunqiu Xu and Chunyu Lin and Bin Xiao},
  year={2026},
  url={https://api.semanticscholar.org/CorpusID:286457517}
}
```

## Meta
If you have any questions about this project, please feel free to drop me an email.

Yuan Mei -- 2551161628@qq.com

