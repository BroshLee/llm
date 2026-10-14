from pydantic import BaseModel, Field


class SkillExtractionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Text from which to extract skills",
    )


class SkillExtractionResult(BaseModel):
    name: str
    role: str
    skills: list[str]


class StructuredJSONResponse(BaseModel):
    model: str
    data: SkillExtractionResult
    usage: dict
    performance: dict