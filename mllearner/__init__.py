"""MLLearner - Comprehensive ML Package for Learners

MLLearner is a high-level machine learning package designed for learners and practitioners.
It provides easy-to-use utilities for:
- Data preprocessing and transformation
- Model building and training
- Evaluation and performance metrics
- Visualization tools
- Helper utilities
"""

__version__ = "0.1.0"
__author__ = "Harsh Tambade"
__email__ = "harsh@example.com"
__license__ = "MIT"

from . import data
from . import models
from . import training
from . import evaluation
from . import visualization
from . import utils

__all__ = [
    "data",
    "models",
    "training",
    "evaluation",
    "visualization",
    "utils",
]
