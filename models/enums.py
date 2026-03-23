from enum import Enum

class DecisionEnum(str, Enum):
    retry = "retry"
    replan = "replan"
    proceed = "proceed"


class CitationStatus(str, Enum):
    valid = "valid"
    stale = "stale"
    broken = "broken"


class SeverityEnum(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ErrorTypeEnum(str, Enum):
    search_failure = "search_failure"
    low_confidence = "low_confidence"
    timeout = "timeout"
    parsing_error = "parsing_error"