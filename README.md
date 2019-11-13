# CE7454-Project-AY19-20
CE7454 Project AY19-20: Kuzushiji Detection with Natural Language Inference

Group 23: Hualin Liu, Anran Hao and Wei Kwek Soh

This repo stores the source code for NTU's CE7454 class project in AY19-20.

The project is based on a kaggle competition of handwritten text extraction from ancient Japanese books.
```
https://www.kaggle.com/c/kuzushiji-recognition/
```
Please download the data following the above link and put them into ```data``` folder

## Background and Motivation
Kuzushiji are a kind of ancient Japanese text which are not comprehensible to modern people. 
By building models to recognize and transcribe them to modern Japanese, we can make more documents accessible which helps us understand ancient Japanese culture and history.

Since there may be syntactic and semantic relations in Kuzushiji texts which are worth mining, 
we would like to leverage language model and conduct several experiments to see if it improves detection results.

## Proposed Method
1. The current 1st place solution (Cascade RCNN + HRNet) is used to extract the high-quality detection proposals.
2. Column clusters are obtained by using unsupervised clustering algorithms (DBScan, Kmeans) over x_center of each proposal.
3. Extract text sequences from columns using reading order (from right to left, up to down).
4. Pretrain language model using ancient Japanese Corpus.
5. Choose proposals of highest confidence for each character, mask those character with confidence score lower than certain threshold


## Project structure


```
├── kaggle-kuzushiji-recognition (Kaggle competition 1st place solution by: https://github.com/tascj/kaggle-kuzushiji-recognition)
│   ├── clustering.py (Script for clustering and extracting sequences)
│   └── ...
├── yolov3 (yolov3 model by: https://github.com/ultralytics/yolov3)
│   └── ...
├── data
│   ├── dtrainval.pkl
│   ├── train_images
│   ├── test_images
│   ├── train_crops
│   └── ...
├── download
│   ├── test_images.zip
│   ├── train_images.zip
│   └── ...
└── submits
    ├── submit001.csv
    └── ...
```
