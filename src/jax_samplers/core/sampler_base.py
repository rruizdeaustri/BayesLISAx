from __future__ import annotations
from typing import Protocol
from .types import PRNGKey
from .result import SamplerResult
from .problem import Problem


class SamplerBase(Protocol):
 def init(self, key: PRNGKey, problem: Problem, **cfg): ...
 def run(self, key: PRNGKey) -> SamplerResult: ...
