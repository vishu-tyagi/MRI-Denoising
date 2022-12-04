# Brain-MRI-Denoising

K-Space Deep Learning Network for Mouse Brain MR Image Denoising

<p align="center">
  <img src="/notebooks/images/input_kspace.png" width="300" height="200" />
  <img src="/notebooks/images/input_mri.png" width="300" height="200" />
</p>

<p align="center">
  <img src="/notebooks/images/target_kspace.png" width="300" height="200" />
  <img src="/notebooks/images/target_mri.png" width="300" height="200" />
</p>

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
python -m sentiment_analysis fetch
```

### Run jupyter server
```
jupyter notebook notebooks/
```

You can now use the jupyter server or `ctorch` kernel to run notebooks.
