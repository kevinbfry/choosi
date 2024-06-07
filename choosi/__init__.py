import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from . import optimizer
from . import choosi_core