# Hybrid UNet-LSTM for Breast Lesion Classification
This repository is an implementation of my master's dissertation "Temporal and Spatial Fusion for Breast Lesion Classification in Ultrasound Videos using Hybrid UNet-LSTM Architecture".

## Abstract
This research focuses on developing a Hybrid UNet-LSTM architecture that enhances the classification of breast lesions in ultrasound videos, crucial for the early and accurate diagnosis of breast cancer. The proposed model integrates Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks, inspired by the U-Net architecture, to capture both spatial and temporal features from video sequences. 

## Requirements
### Setting Up the Conda Environment

To ensure that all dependencies are properly managed, it is recommended to use a Conda environment. Follow these steps:

1. **Create the Conda Environment:**

    ```bash
    conda create -n cva_net python tensorflow scikit-learn -y
    ```

2. **Activate the Environment:**

    ```bash
    source activate cva_net
    ```

3. **Install Additional Dependencies:**

    After activating the environment, install the remaining dependencies:

    ```bash
    conda install -c conda-forge opencv matplotlib -y
    ```

4. **Verify the Installation:**

    Ensure all necessary packages are installed by running:

    ```bash
    conda list
    ```

    This should display a list of all the installed packages.


## Dataset

The dataset used for this research is the CVA-Net dataset (Lin et al., 2022), which can be downloaded from the following link:

[Download CVA-Net Dataset](https://github.com/jhl-Det/CVA-Net)

The CVA-Net dataset and baseline model were introduced in the following paper:

> Lin, Z., Lin, J., Zhu, L., Fu, H., Qin, J., & Wang, L. (2022). A New Dataset and a Baseline Model for Breast Lesion Detection in Ultrasound Videos. *Medical Image Computing and Computer Assisted Intervention -- MICCAI 2022*, Springer, Cham, 614-623.

Please consider citing the original CVA-Net paper if you use this dataset in your research.

### Dataset Directory Structure

```
Dataset/
│
├── benign/
│   ├── 2cda21c3aab26332/
│   │   ├── 000000.png
│   │   ├── 000001.png
│   │   └── ...
│   └── ...
└── malignant/
    ├── 1dc9ca2f1748c2ec/
    │   ├── 000000.png
    │   ├── 000001.png
    │   └── ...
    └── ...
```

## Trained Models

Pre-trained models can be downloaded from the following Google Drive link:

[Download Pre-trained Models](https://drive.google.com/drive/folders/1IbXMqSxZHMv1nV88xg-A6u0tGOwJF2BE?usp=share_link)


## How to Reproduce the Experiment

### 1. Clone the Repository

```bash
git clone https://github.com/sakthi-yazhini/HybridUNet-LSTM.git
cd HybridUNet-LSTM
```

### 2. Setup the Conda Environment

Ensure you have Conda installed on your machine, then create and activate the environment as outlined in the [Requirements](#requirements) section.

### 3. Prepare the Dataset

Download the dataset from the provided link and place it in the `Dataset/` directory, following the structure mentioned above.

### 4. Train and Evaluate the Models

Each script provided in this repository handles both the training and evaluation of the models. After training, the models are automatically evaluated on the test set, and the evaluation metrics, such as accuracy, precision, recall, F1-score, and others, will be printed directly in the console.

You can train and evaluate the models by running the respective Python scripts. For example:

- **Train and Evaluate the CNN Model:**

    ```bash
    python cnn.py
    ```

- **Train and Evaluate the Transformer Model:**

    ```bash
    python transformer.py
    ```

- **Train and Evaluate the Hybrid UNet-LSTM Model:**

    ```bash
    python unet_lstm.py
    ```

These scripts will handle the entire process from loading the dataset, training the model, and evaluating its performance. The results, including confusion matrices, classification reports, and other relevant metrics, will be displayed in the console after the evaluation is complete.

## Running on HPC (ARC4)

For users who wish to run the models on a High Performance Computing (HPC) cluster like ARC4, shell scripts are provided to submit the jobs.

### Submitting Jobs to ARC4

1. **Modify the Shell Script:**

    Edit the provided `.sh` shell script to match your environment's specifics, such as paths, resource allocation, and module loading.

2. **Submit the Job:**

    Submit the job to ARC4 using the following command:

    ```bash
    qsub cnn_job.sh
    ```

    ```bash
    qsub transformer_job.sh
    ```

    ```bash
    qsub unet_lstm_job.sh
    ```



3. **Monitor the Job:**

    You can monitor your job status with:

    ```bash
    qstat
    ```

### Acknowledgment

This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK. 
