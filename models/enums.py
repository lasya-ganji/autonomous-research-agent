from enum import Enum


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
    system_error = "system_error"
    budget_exceeded = "budget_exceeded"
    loop_limit = "loop_limit"
    api_error = "api_error"
    timeout_error = "timeout_error"
    network_error = "network_error"
    unknown_error = "unknown_error"