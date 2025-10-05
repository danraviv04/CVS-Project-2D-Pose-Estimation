# 2D Pose Estimation for Surgical Instruments

## Overview
This project implements a complete pipeline for **2D pose estimation of surgical tools** using synthetic data generation, deep learning, and unsupervised refinement on unlabeled surgical videos.  
The system focuses on **tweezers** and **needle holders**, and aims to generalize from synthetic imagery to real surgical videos, addressing the **domain gap** between simulation and reality.

---

## Table of Contents
- [1. Project Phases](#1-project-phases)
- [2. Repository Structure](#2-repository-structure)
- [3. Environment Setup](#3-environment-setup)
- [4. Phase 1 — Synthetic Data Generation](#4-phase-1--synthetic-data-generation)
- [5. Phase 2 — Model Training](#5-phase-2--model-training)
- [6. Phase 3 — Unsupervised Refinement](#6-phase-3--unsupervised-refinement-on-real-videos)
- [7. Deliverables](#7-deliverables)

---

## 1. Project Phases

| Phase | Goal | Description | 
|-------|------|-------------|
| 1 | Synthetic data generation | Generate a labeled dataset (1000 images) using BlenderProc. Includes lighting, background, and articulation variability. |
| 2 | 2D pose estimation model | Train a YOLO-style Segmentation model on the synthetic dataset for instrument detection and pose estimate based on predictions. |
| 3 | Unsupervised refinement | Improve real-world generalization using pseudo-labeling and self-training on unlabeled surgical videos. |

---

## 2. Repository Structure

```
CVS-Project-2D-Pose-Estimation/
├── data_generation/
│   ├── annotations_to_coco.py
│   ├── coco_to_yolo_seg.py
│   ├── generate_tools.py
│   ├── make_yolo_pose_from_keypoints.py
│   ├── paste_on_random_background.py
│   ├── place_coordinates_txt.py
│   └── visualize_coco_polygons.py
│
├── models/
│   ├── synthetic_weights.pt
│   └── refined_weights.pt
│
├── hands/
│   └── hand_1.obj              # right hand object
│
├── curate_and_split.py         # Curates and splits datasets into train/val
├── model_refine.py             # Fine-tunes pretrained model on pseudo-labeled real data
├── predict.py                  # Inference on images
├── synthetic_data_generator.py # Main entry point for synthetic dataset generation
├── train.py                    # YOLO-style model training
├── video.py                    # Inference and visualization on surgical videos
├── requirements.txt
└── README.md
```

---

## 3. Environment Setup

### Installation
```bash
git clone https://github.com/danraviv04/CVS-Project-2D-Pose-Estimation.git
cd CVS-Project-2D-Pose-Estimation

# Create and activate a Python environment (Python 3.10 recommended)
conda create -n cvs_pose python=3.10 -y
conda activate cvs_pose

# Install dependencies
pip install -r requirements.txt

```

### Notes
- The requirements.txt file already includes all necessary packages for:
  - BlenderProc — synthetic data generation and 3D rendering.
  - PyTorch / Ultralytics — training YOLO-based pose estimation models.
  - Albumentations / Albucore — image augmentation utilities.
- Blender ≥ 4.0 is recommended for rendering high-quality synthetic data.
- For GPU-based rendering or training, ensure that CUDA and PyTorch GPU builds are properly installed.

---

## 4. Phase 1 — Synthetic Data Generation

### Goal
Create a labeled dataset of ≥1000 synthetic surgical tool images with 2D keypoints and bounding boxes, simulating realistic operating-room variability.

### Features Implemented
- **Instrument positioning/orientation:** randomized pose and rotation.
- **Lighting variation:** random intensity, color temperature, and direction.
- **Background variation:** random real-image or HDRI environment maps.
- **Additional creative variations:**
  - (1) *Articulation control:* dynamic open/close states of jaws.
  - (2) *Material perturbation:* metallic roughness and color-tone jitter for realism.

### Reproduce
```bash
python synthetic_data_generator.py --num_images 1000 --debug
```

### Output
```
output/
├── coco_data/                     # COCO-style dataset
│   ├── images/                    # No Background data
│   ├── annotations_train.json
│   ├── annotations_val.json
│   └── coco_annotations.json      
│
├── composited/
│   ├── train/                     # train composites (RGB + alpha)
│   └── val/                       # validation composites
│
├── keypoints/                     # Per-image keypoint JSONs
│
├── yolo_data/                     # YOLO keypoint-format dataset
│   ├── images/
│   ├── labels/
│   └── data.yaml                  # YOLO config for pose training
│
└── yolo_data_seg/                 # YOLO segmentation-format dataset
    ├── images/
    ├── labels/
    ├── viz_train/                 # sample visualizations
    ├── viz_val/
    └── data.yaml                  # YOLO config for segmentation training
```

---

## 5. Phase 2 — Model Training

### Goal
Train a YOLO-based 2D keypoint model using the synthetic dataset.

### Reproduce
```bash
    python train.py \
    --data /home/student/project/output/yolo_data_seg/data.yaml \
    --model yolo11s-seg.pt \
    --epochs 90 --imgsz 1536 --batch -1 --device 0 \
    --project seg_phaseB --name yolo11s_seg_or_aug_2super_orheavy \
    --warmup_epochs 3 --base_strength 0.70 --max_strength 1.20 --clahe_clip 2.0 \
    --derive_on none --derive_imgsz 1536 --derive_iou 0.55 \
    --scales 1.25,1.0 --conf_ladder 0.36,0.32,0.28
```

### Outputs
```
seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/
├── weights/
│   ├── best.pt
│   └── last.pt
├── results.csv
.
.
.
└── results.png
```

### Evaluation
Qualitative evaluation is performed on the provided **real surgical video (`4_2_24_A_1.mp4`)**, visualizing detection and pose estimation results.

```bash
 python video.py \
--weights /home/student/project/seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt \
--video /datashare/project/vids_test/4_2_24_A_1.mp4 \
--out_dir out_vis/final_pred --imgsz 1536 --conf 0.35 --iou 0.55 --device 0 --max_det 8 \
--retina_masks --draw_kpts --progress_every 1 --pre_clahe --clahe_clip 2.0
```

---

## 6. Phase 3 — Unsupervised Refinement on Real Videos

### Goal
Refine the model using unlabeled real videos via **self-training (pseudo-labeling)**.

### Reproduce

#### Step 1 — Generate Pseudo-Labels
You can use the Evaluation `video.py` call used in the Evaluation from the phase above (make sure to use it on the `vids_tune/4_2_24_B_2.mp4` and `vids_tune/20_2_24_1.mp4` videos)

#### Step 2 — Curate and create the pseudo-labels' dataset
clean the predictions and take the best ones
```bash
python curate_and_split.py \
  --pair /datashare/project/vids_tune/4_2_24_B_2.mp4::out_vis/ultra_like_4_2_24_B_2/labels \
  --pair /datashare/project/vids_tune/20_2_24_1.mp4::out_vis/ultra_like_20_2_24_1/labels \
  --out_ds /home/student/project/pseudo_ultra_ds \
  --min_conf_nh 0.50 --min_conf_t 0.40 \
  --stride 1 --require_any \
  --save_frames --jpg_quality 92 \
  --split 0.9 --split_by frame --seed 0 \
  --progress_every 50 
```
#### Step 3 — Retrain the model
use **`refine.py`**
```bash
python refine.py \
--data /home/student/project/pseudo_ultra_ds/data.yaml \
--base_weights /home/student/project/seg_phaseB/yolo11s_seg_or_aug_2super_orheavy/weights/best.pt \
--epochs 50 --topup_epochs 12 \
--self_train --self_min_conf_nh 0.55 --self_min_conf_t 0.45 \
--imgsz 1536 --device 0 --project seg_phaseB --name refined_seg 
```

#### Step 4 - Inference
just replace the seg_phaseB subfolder to your refines model
```bash
python video.py \
--weights /home/student/project/seg_phaseB/refined_seg_topup/weights/best.pt \
--video /datashare/project/vids_test/4_2_24_A_1_small.mp4 \
--out_dir out_vis/final_pred \
--imgsz 1536 --conf 0.35 --iou 0.55 --device 0 --max_det 8 \
--retina_masks --draw_kpts --progress_every 1 \
--pre_clahe --clahe_clip 2.0
```

---

## 7. Deliverables

### Download Links
| Model | Path / Link |
|--------|--------------|
| Synthetic-only model | [synthetic weights](models/synthetic_weights.pt) |
| Refined model (SSL) | [refined weights](models/refined_weights.pt) |
