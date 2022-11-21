from ctorch.utils.constants import (
    IN_CHANNELS,
    OUT_CHANNELS,
    HIDDEN_DIMENSIONS,
    PARAMETERS,
    CONV2D,
    MAXPOOL2D,
    CONVTRANSPOSE2D,
    OUTCONV2D
)


class ComplexTorchConfig():
    # S3 bucket for fetching raw data
    S3_BUCKET = "kspace-mri"
    CURRENT_PATH = None   # will be set to working directory by os.getcwd()

    # Files to download from S3 bucket
    S3_BUCKET_RELEVANT_FILES = ["Train.zip", "Val.zip", "Test.zip"]

    # Arrays of shape (4, 18, HEIGHT, WIDTH) will be padded to
    # obtain the shape (4, 18, NEW_HEIGHT, NEW_WIDTH)
    PREPROCESSING_NEW_HEIGHT = 208
    PREPROCESSING_NEW_WIDTH = 416

    # Model
    MODEL_PARAMETERS = {
        IN_CHANNELS: 4,
        OUT_CHANNELS: 4,
        HIDDEN_DIMENSIONS: [32, 64, 128, 256],
        PARAMETERS: {
            CONV2D: {"kernel_size": 3, "stride": 1, "padding": 1},
            MAXPOOL2D: {"kernel_size": 2},
            CONVTRANSPOSE2D: {"kernel_size": 2, "stride": 2, "padding": 0},
            OUTCONV2D: {"kernel_size": 1, "stride": 1, "padding": 0}
        }
    }
