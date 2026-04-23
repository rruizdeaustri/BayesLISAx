from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Any, Optional

@dataclass
class SamplerResult:
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    #extra: Dict[str, Any] = field(default_factory=dict)
