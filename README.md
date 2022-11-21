# Brain-MRI-Denoising

K-Space Deep Learning Network for Mouse Brain MR Image Denoising

## Setup to run jupyter notebooks

### Move into top-level directory
```
cd Brain-MRI-Denoisingn
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
pip install -e src/ctorch
```
Including the optional -e flag will install package in "editable" mode, meaning that instead of copying the files into your virtual environment, a symlink will be created to the files where they are.

You can now use the jupyter kernel `ctorch` to run notebooks.
