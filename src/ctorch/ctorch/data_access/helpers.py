import os
import logging
import logging.config
from pathlib import Path
import zipfile

import boto3

from ctorch.utils import timing

logger = logging.getLogger(__name__)


@timing
def s3_download(s3_bucket: str, file: str, out_path: str):
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"]
    )
    s3.download_file(s3_bucket, file, out_path)


@timing
def unzip(file_path: Path, out_path: Path) -> None:
    """
    Unzip a zip file
    Args:
        file_path (Path): Path to the zip file
        out_path (Path): Path for the extracted files
    """
    with zipfile.ZipFile(file_path, "r") as zip:
        zip.extractall(out_path)
    return
