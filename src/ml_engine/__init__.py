"""ML Engine - World-Class Machine Learning Pipeline."""

__version__ = "0.1.0"

from ml_engine.utils.logger import setup_logging
from ml_engine.utils.exceptions import MLEngineException

setup_logging()

__all__ = ["setup_logging", "MLEngineException"]
