import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from ctorch.config import ComplexTorchConfig
from ctorch.data_access.helpers import s3_download, unzip
from ctorch.utils.constants import (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    MODEL_DIR,
    SAVED_MODELS_DIR,
    REPORTS_DIR,
    TRAIN,
    VAL,
    TEST,
    INPUT,
    TARGET
)

logger = logging.getLogger(__name__)


class DataClass():
    def __init__(self, config: ComplexTorchConfig):
        self.config = config
        self.s3_bucket = config.S3_BUCKET

        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))
        self.model_path = Path(os.path.join(self.current_path, MODEL_DIR))
        self.reports_path = Path(os.path.join(self.current_path, REPORTS_DIR))

    def make_dirs(self):
        dirs = [
            Path(os.path.join(self.data_path, RAW_DIR)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, TRAIN, INPUT)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, TRAIN, TARGET)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, VAL, INPUT)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, VAL, TARGET)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, TEST, INPUT)),
            Path(os.path.join(self.data_path, PROCESSED_DIR, TEST, TARGET)),
            Path(os.path.join(self.model_path, SAVED_MODELS_DIR)),
            self.reports_path
        ]
        for dir in dirs:
            dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created data directory {self.data_path}")
        logger.info(f"Created model directory {self.model_path}")
        logger.info(f"Created reports directory {self.reports_path}")

    def _fetch(self):
        for file in self.config.S3_BUCKET_RELEVANT_FILES:
            out_path = os.path.join(self.data_path, RAW_DIR, file)
            logger.info(f"Downloading s3://{self.s3_bucket}/{file}")
            s3_download(self.s3_bucket, file, out_path)

    def _unzip(self):
        for file in self.config.S3_BUCKET_RELEVANT_FILES:
            out_path = Path(os.path.join(self.data_path, RAW_DIR))
            file_path = Path(os.path.join(out_path, file))
            logger.info(f"Unzipping {file_path}")
            unzip(file_path, out_path)

    def build(self):
        self._fetch()
        self._unzip()
        logger.info(
            f"Raw data available for preprocessing {os.path.join(self.data_path, RAW_DIR)}"
        )


class KSpaceDataset(Dataset):
    def __init__(self, input_path: Path, target_path: Path):
        super().__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.files = sorted(os.listdir(self.input_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        X = np.load(os.path.join(self.input_path, file))
        Y = np.load(os.path.join(self.target_path, file))
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        return X, Y
