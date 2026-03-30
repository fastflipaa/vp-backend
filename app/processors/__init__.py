"""State processors for the LEVITAS conversation pipeline.

Each processor handles a specific conversation state (Greeting, Qualifying,
Handoff, Scheduling) and returns a ProcessorResult with the AI response,
state transition, and metadata.

BaseProcessor defines the interface. ProcessorResult is the return type.
"""

from app.processors.base import BaseProcessor, ProcessorResult
from app.processors.greeting import GreetingProcessor
from app.processors.handoff import HandoffProcessor
from app.processors.qualifying import QualifyingProcessor
from app.processors.scheduling import SchedulingProcessor

__all__ = [
    "BaseProcessor",
    "ProcessorResult",
    "GreetingProcessor",
    "HandoffProcessor",
    "QualifyingProcessor",
    "SchedulingProcessor",
]
