from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


@dataclass
class SamplerResult:
    samples: np.ndarray
    weights: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Optionally persist a compact posterior bundle for workflow afterburners.

        Saving is opt-in and controlled by ``JAX_SAMPLERS_POSTERIOR_PATH``. This
        keeps sampler APIs unchanged while allowing Snakemake jobs to retain the
        complete physical posterior instead of relying only on parsed log output.
        """
        output = os.environ.get("JAX_SAMPLERS_POSTERIOR_PATH", "").strip()
        if not output:
            return

        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.asarray(self.samples)
        weights = np.empty(0, dtype=float) if self.weights is None else np.asarray(self.weights)
        diagnostics_json = json.dumps(self.diagnostics or {}, default=_json_default, sort_keys=True)
        np.savez_compressed(
            path,
            samples=samples,
            weights=weights,
            diagnostics_json=np.asarray(diagnostics_json),
            seed=np.asarray(os.environ.get("JAX_SAMPLERS_POSTERIOR_SEED", "")),
            algo=np.asarray(os.environ.get("JAX_SAMPLERS_POSTERIOR_ALGO", "")),
            config_path=np.asarray(os.environ.get("JAX_SAMPLERS_CONFIG", "")),
        )
