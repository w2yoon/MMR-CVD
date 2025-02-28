# Multimodal Retinal Imaging for CVD Prediction

## Project Overview
Cardiovascular diseases (CVDs) are one of the leading global health challenges, and early, accurate risk assessment is critical for effective intervention. Retinal imaging has emerged as a promising non-invasive method for assessing systemic vascular health, as retinal features are strongly linked to CVD risk. However, traditional approaches relying solely on Color Fundus Photography (CFP) provide only 2D views and may overlook subtle microvascular abnormalities. In contrast, Optical Coherence Tomography (OCT) offers high-resolution, 3D structural information but remains underutilized. To address these gaps, our project introduces a novel multi-modal framework that integrates both CFP and OCT images. The model employs a Bidirectional Cross-Attention mechanism to fuse complementary features from both modalities, uses fine-tuning for vascular feature extraction from CFP, and leverages a 3D CNN module to capture depth-related structural details. Additionally, our multi-label classification approach allows for the simultaneous prediction of multiple cardiovascular conditions, delivering a more comprehensive and clinically relevant risk assessment. Experimental results indicate that our approach outperforms existing methods in terms of accuracy and F1-score.

## Project Structure
```bash
├── data
│   ├── test.csv
│   └── test_retinal_img
│       ├── cfp
│       └── oct
└── code
    ├── dataset.py
    ├── utils.py
    ├── model.py
    ├── train.py
    └── test.py
```

## Pretrained Weights
You can download pretrained weights here.

## Training and Evaluation
The project is designed to be user-friendly, allowing you to train and evaluate the model directly via command-line arguments.

### Training
To train the model, run the following command:
```bash
python train.py --csv_path /path/to/train.csv --cfp_dir /path/to/cfp --oct_dir /path/to/oct --epochs 150 --batch_size 16 --num_workers 8 --save_path model_checkpoint.pth
```

This command will:

* Load the dataset based on the provided CSV file and image directories.
* Apply appropriate image transformations for CFP and OCT images.
* Train the multi-modal fusion network using the specified hyperparameters.
* Save the trained model to the designated checkpoint file.

### Evaluation

#### Test Dataset Preparation
For testing, please download the test image dataset from here. Once downloaded, make sure that the **cfp** and **oct** folders are placed inside the **/data/test_retinal_img** directory.

To evaluate the model, run:
```bash
python test.py --csv_path /path/to/test.csv --cfp_dir /path/to/cfp --oct_dir /path/to/oct --model_path model_checkpoint.pth --batch_size 16 --num_workers 8
```

* The evaluation script computes performance metrics including accuracy and F1-score for each disease class.


