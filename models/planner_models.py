from pydantic import BaseModel, Field

class PlanStep(BaseModel):
    step_id: int
    question: str = Field(min_length=5)
    priority: int = Field(ge=1, le=5)