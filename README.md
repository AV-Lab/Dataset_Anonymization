# README.md

# 1. 🎥 Video Anonymization with EgoBlur

This repository provides a pipeline to anonymize video datasets by blurring sensitive content (e.g., license plates, faces) using the **EgoBlur** model.

## 🔎 What is This?
With increasing privacy concerns and legal restrictions on visual data, it's essential to anonymize personally identifiable information in videos. This project uses **EgoBlur**, a vision model for responsible anonymization, to automatically detect and blur such information in a scalable and smart way.

## 🤯 Key Features
- License plate detection and elliptical Gaussian blurring
- Object ID tracking across frames using annotations
- Adaptive multi-pass thresholding for robust detection
- Dynamic cropping thresholds to skip tiny irrelevant regions

## 📅 EgoBlur: Responsible Innovation in Aria
Cited from:
> Raina et al., "EgoBlur: Responsible Innovation in Aria", arXiv:2308.13093 ([link](https://arxiv.org/abs/2308.13093))

## 🚀 Installation (Skip Mobile SAM if verification GUI is not desired)
```bash
git clone https://github.com/your-username/egoblur-anonymization
cd egoblur-anonymization
conda create -n ego_blur python=3.9
conda activate ego_blur
cd src
git clone https://github.com/ChaoningZhang/MobileSAM.git
pip install -e MobileSAM
pip install -r requirements.txt
```

Make sure to place the model checkpoint `ego_blur_lp.jit` under `models/`.

## 🌐 Directory Structure
```
.
├── src/
│   ├── main_pipeline_with_cropping.py
│   ├── ego_blur_utils.py
│   └── ego_blur_utils_eliptical.py
├── frames/
│   └── video_220047/   # extracted frames
├── models/
│   └── ego_blur_lp.jit
├── output_annotation_crop_fast/
├── annotations/
│   └── tracking_annotations/gmot/video_220047.txt
```

## ⚖️ Pipeline Overview
![Flowchart](docs/Flowchart.jpg)


## 🔢 How It Works
1. **For each frame in the video**:
   - Get all bounding boxes and object IDs from the annotation file
   - Crop the bounding box region
   - If area is too small, skip
   - Try to blur it using EgoBlur (up to 5 attempts with decreasing threshold)
   - Replace in frame if blurred, otherwise log
2. Save updated frame
3. Export log as JSON summarizing blur status of each object across frames

## 💡 Tips
- You can tune `min_crop_area_ratio` and `nms_iou_threshold` for different datasets
- Blurring logic is separated in `ego_blur_utils_eliptical.py` and can be modified for different blur effects

## 🎥 Sample Results
> Before vs After blurring visualizations and sample video outputs

Demo
![Demo](media/demo.gif)


# 2. Manual Verification User Interface
After processing the frames automatically using EgoBlur, this blurring interfance that utilizes MobileSAM can be used for manual verification of blurred frames.

---

## Table of Contents

1. [Features](#features)
2. [Requirements](#requirements)
3. [Directory Structure](#directory-structure)

## Features

- **Point-based blur**: Click on the image to apply a blur within the SAM-predicted mask.
- **Rectangular blur**: Hold `Shift` and drag to select a rectangular region for blurring.
- **Undo/Redo**: Quickly reverse or reapply edits via buttons or `Ctrl+Z` / `Ctrl+Y`.
- **Image checklist**: View saved/unsaved status and jump to any image in the sequence.
- **Keyboard navigation**: Use ←/→ to move between images.
- **Saving Images**: Click save and next to save your changes.
- **Laucnhing the app**: Use the following code to launch the app:
```bash
cd src
python sam_blur_gui.py
```

To disable SAM clicks, use:
```bash
cd src
python sam_blur_gui.py --disable-sam
```


## Directory Structure 
This can be used separate from the previous pipeline in case manual blurring is desired. Data should be structured as follows:
```
.
├── data/
│   ├── raw/ # raw extracted frames
│   ├── verify/ # blurred frames from EgoBlur pipeline. Place raw images here in case of manual blurring.
│   ├── final/ # final blurred frames
├── src/
│   └── sam_blur_gui.py/   # extracted frames
```
## 🔗 Cite Our Paper
```bibtex
@misc{madjid2025emtvisualmultitaskbenchmark,
      title={EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region}, 
      author={Nadya Abdel Madjid and Murad Mebrahtu and Abdelmoamen Nasser and Bilal Hassan and Naoufel Werghi and Jorge Dias and Majid Khonji},
      year={2025},
      eprint={2502.19260},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.19260}, 
}
```
## 🔗 References
```bibtex
@misc{raina2023egoblurresponsibleinnovationaria,
      title={EgoBlur: Responsible Innovation in Aria}, 
      author={Nikhil Raina and Guruprasad Somasundaram and Kang Zheng and Sagar Miglani and Steve Saarinen and Jeff Meissner and Mark Schwesinger and Luis Pesqueira and Ishita Prasad and Edward Miller and Prince Gupta and Mingfei Yan and Richard Newcombe and Carl Ren and Omkar M Parkhi},
      year={2023},
      eprint={2308.13093},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2308.13093}, 
}
```
```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```
---
Developed with ❤️ for responsible AI research.
