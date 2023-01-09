# Brain-MRI-Denoising

Complex K-Space Deep Learning Network for Mouse Brain MR Image Denoising

<p align="center">
  <img src="/notebooks/images/input_k_space.png" width="300" height="200" />
  <img src="/notebooks/images/input_mr_image.png" width="300" height="200" />
</p>

<p align="center">
  <img src="/notebooks/images/target_k_space.png" width="300" height="200" />
  <img src="/notebooks/images/target_mr_image.png" width="300" height="200" />
</p>

This is Complex k-space UNet, a variant of the UNet architecture specifically designed for processing complex-valued data, such as data from magnetic resonance imaging (MRI) scans. In MRI, the raw data is collected in the k-space domain, which is a complex-valued representation of the spatial frequency information in the image. The complex k-space UNet is trained to take as input a complex-valued k-space image and output a segmentation mask in the image space.

Data was acquired from Professor Jia Guo at Columbia University

Total number of trainable parameters: 1,925,988

**Loss curve**

<p align="center">
  <img src="/notebooks/images/loss_curve.png" width="400" height="300" />
</p>

**Evaluation curves**

<p align="center">
  <img src="/notebooks/images/psnr_curve.png" width="350" height="250" />
    <img src="/notebooks/images/pcc_curve.png" width="350" height="250" />
</p>

<p align="center">
  <img src="/notebooks/images/ssim_curve.png" width="350" height="250" />
  <img src="/notebooks/images/scc_curve.png" width="350" height="250" />
</p>

**Test Results**

<p align="center">
  <img src="/notebooks/images/test_results.png" width="400" height="100" />
</p>

The table above displays results on test dataset consisting of 1008 samples. For all evaluation metrics, we observe that the similarity scores increased. This means the predicted MR images are closer to the ground truth than the input images. Hence, the model succeeded in denoising the input images.

Here, PSNR: Peak Signal to Noise Ratio, PCC: Pearson Correlation Coefficient, SSIM: Structural Similarity Index, SCC: Spearman Correlation Coefficient

**Notebooks**

The notebooks may be viewed in the following order:

1. *[explore-data.ipynb](notebooks/explore-data.ipynb)* - Explore data and visualize MR Images

2. *[unet-train.ipynb](notebooks/unet-train.ipynb)* - Train UNet

3. *[inference.ipynb](notebooks/inference.ipynb)* - Inference on test dataset using the trained model

**Future Work**

Benchmark against other approaches

## Setup Instructions

### Move into top-level directory
```
cd MRI-Denoising
```

### Install environment
```
conda env create -f environment.yml
```

### Activate environment
```
conda activate ctorch
```

### Install package
```
pip install -e src/complex-torch
```
Including the optional -e flag will install package in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

### Fetch data
```
python -m ctorch fetch
```

Requires AWS credentials. Please email me vt2353@columbia.edu for access.

### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter server or `ctorch` kernel to run notebooks.
