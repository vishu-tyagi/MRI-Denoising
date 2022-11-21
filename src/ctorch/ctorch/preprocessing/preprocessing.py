import os
import logging
from pathlib import Path

import numpy as np

from ctorch.config import ComplexTorchConfig
from ctorch.utils import timing
from ctorch.utils.constants import (
    DATA_DIR,
    RAW_DIR,
    PROCESSED_DIR,
    TRAIN,
    VAL,
    TEST,
    INPUT,
    TARGET,
    DIM_COIL,
    DIM_HEIGHT,
    DIM_WIDTH,
    NUM_SLICE,
    HEIGHT,
    WIDTH
)

logger = logging.getLogger(__name__)


class Processor():
    def __init__(self, config: ComplexTorchConfig):
        self.config = config
        self.current_path = Path(os.getcwd()) if not config.CURRENT_PATH else config.CURRENT_PATH
        self.data_path = Path(os.path.join(self.current_path, DATA_DIR))

    def _calculate_pad(self) -> list[tuple[int, int]]:
        H_difference = self.config.PREPROCESSING_NEW_HEIGHT - HEIGHT
        W_difference = self.config.PREPROCESSING_NEW_WIDTH - WIDTH
        H_pad = (round(H_difference / 2), H_difference - round(H_difference / 2))
        W_pad = (round(W_difference / 2), W_difference - round(W_difference / 2))
        pad = [(0, 0), (0, 0), H_pad, W_pad]
        return pad

    @timing
    def preprocess(self, split, source):
        pad = self._calculate_pad()
        from_path = os.path.join(self.data_path, RAW_DIR, split, source)
        out_path = os.path.join(self.data_path, PROCESSED_DIR, split, source)
        logger.info(f"Processing files in {os.path.join(split, source)}")
        for file_name in os.listdir(from_path):
            X = np.load(os.path.join(from_path, file_name)).astype(np.csingle)
            X_max = (np.abs(X)).max(axis=(DIM_COIL, DIM_HEIGHT, DIM_WIDTH))
            X_max = np.expand_dims(X_max, (DIM_COIL, DIM_HEIGHT, DIM_WIDTH))
            X = np.divide(X, X_max)
            X = np.pad(array=X, pad_width=pad, mode="constant")
            for slice in range(NUM_SLICE):
                X_sliced = X[:, slice, :, :]
                out_file_name = file_name.split(".")[0] + f"_{slice:02}.npy"
                np.save(os.path.join(out_path, out_file_name), X_sliced)
        logger.info(f"Processed {os.path.join(split, source)} files saved to {out_path}")
        return self

    def build(self):
        for split in [TRAIN, VAL, TEST]:
            for source in [INPUT, TARGET]:
                self.preprocess(split, source)
            input = os.listdir(os.path.join(self.data_path, PROCESSED_DIR, split, INPUT))
            target = os.listdir(os.path.join(self.data_path, PROCESSED_DIR, split, TARGET))
            input = sorted(input)
            target = sorted(target)
            assert (input == target)
