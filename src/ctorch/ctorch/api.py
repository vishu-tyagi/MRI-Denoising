import logging

from ctorch.config import ComplexTorchConfig
from ctorch.data_access import DataClass
from ctorch.preprocessing import Processor
from ctorch.utils import timing

logger = logging.getLogger(__name__)


@timing
def fetch(config: ComplexTorchConfig = ComplexTorchConfig) -> None:
    logger.info("Fetching raw data...")
    data = DataClass(config)
    data.make_dirs()
    data.build()

    logger.info("Processing raw data...")
    processor = Processor(config)
    processor.build()
    return
