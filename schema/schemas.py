from pydantic import BaseModel, Field


class OutputSchema(BaseModel):
    candidate_name: str = Field(description="This is the candidate name from the resume", default="")
    contact_number: int = Field(description="This is the contact number of the candiate from the resume", default=0)
    email_id: str = Field(description="This is the email id of the candidate from the resume", default="")
    skills: list[str] = Field(description="This is the list of skills of the candidate from the resume", default=[])


