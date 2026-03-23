from pydantic import BaseModel
from models.enums import SeverityEnum, ErrorTypeEnum

class ErrorLog(BaseModel):
    node: str
    timestamp: str
    severity: SeverityEnum
    type: ErrorTypeEnum
    message: str