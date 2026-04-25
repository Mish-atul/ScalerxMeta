"""CourtLLM Environment - Multi-agent courtroom for LLM hallucination reduction"""

from .models import CourtAction, CourtObservation, CourtState
from .client import CourtLLMClient

__version__ = "1.0.0"
__all__ = ["CourtAction", "CourtObservation", "CourtState", "CourtLLMClient"]
