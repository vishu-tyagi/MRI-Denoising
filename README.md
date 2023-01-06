# Brain-MRI-Denoising

Complex K-Space Deep Learning Network for Mouse Brain MR Image Denoising

<p align="center">
  <img src="/notebooks/images/input_kspace.png" width="300" height="200" />
  <img src="/notebooks/images/input_mri.png" width="300" height="200" />
</p>

<p align="center">
  <img src="/notebooks/images/target_kspace.png" width="300" height="200" />
  <img src="/notebooks/images/target_mri.png" width="300" height="200" />
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
- Peak Signal to Noise Ratio
    - Input vs Ground Truth: 37.48
    - Predicted vs Ground Truth: 38.013
- Pearson Correlation Coefficient
    - Input vs Ground Truth: 0.9726
    - Predicted vs Ground Truth: 0.9804
- Structural Similarity Index
    - Input vs Ground Truth: 0.9346
    - Predicted vs Ground Truth: 0.9617
- Spearman Correlation Coefficient
    - Input vs Ground Truth: 0.8738
    - Predicted vs Ground Truth: 0.8962

In all cases, we see that the similarity scores increased. This means the predicted MR images are closer to the ground truth than the input images. Hence, the model succeeded in denoising the input images.

**Future Work**

Benchmark this model against other approaches

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

### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter server or `ctorch` kernel to run notebooks.
