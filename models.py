from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActionTriple:
    subject: str
    verb: str
    object: str
    sent_id: int
