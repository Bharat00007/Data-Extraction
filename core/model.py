from dataclasses import dataclass
from typing import Optional

@dataclass
class FieldResult:
    value: Optional[str]
    confidence: float
    source: str
    reason: str
