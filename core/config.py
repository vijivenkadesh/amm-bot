from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator




class EnvSettings(BaseSettings):
    OPENAI_API_KEY: str = Field(description="This is API key for Open AI",
                                default="",
                                strict=True)
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_api(cls, v: str):
        prefix = "sk-"
        if not v.startswith(prefix):
            raise ValueError(f"API key must start with '{prefix}'")
        return v


try:
    settings = EnvSettings()
except ValueError as e:
    print(f"This is the error {str(e)}")
finally:
    print("Please set the API key properly")
