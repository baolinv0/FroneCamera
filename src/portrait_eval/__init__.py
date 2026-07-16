"""Front-camera portrait evaluation system."""

from __future__ import annotations

import sys

from portrait_eval import repository_v2 as _repository_v2_base
from portrait_eval import repository_v2_integrity as _repository_v2_integrity

sys.modules[f"{__name__}.repository_v2"] = _repository_v2_integrity

__version__ = "0.1.0"

__all__ = ["__version__"]
