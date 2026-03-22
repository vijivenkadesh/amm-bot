from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from pydantic import SecretStr
from typing import Optional
import os




class EnvSettings(BaseSettings):
    OPENAI_API_KEY: Optional[SecretStr] = Field(description="This is API key for Open AI",
                                default=None,
                                strict=True)
    LANGSMITH_TRACING: str = "true"
    LANGSMITH_ENDPOINT: str = ""
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = ""
    MODEL: str = ""
    

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_api(cls, v: SecretStr):
        prefix = "sk-"
        if not v.get_secret_value().startswith(prefix):
            raise ValueError(f"API key must start with '{prefix}'")
        return v
    
    def load_langsmith(self):
        os.environ['LANGSMITH_TRACING'] = self.LANGSMITH_TRACING
        os.environ['LANGSMITH_ENDPOINT'] = self.LANGSMITH_ENDPOINT
        os.environ['LANGSMITH_API_KEY'] = self.LANGSMITH_API_KEY
        os.environ['LANGSMITH_PROJECT'] = self.LANGSMITH_PROJECT


try:
    settings = EnvSettings()
    settings.load_langsmith()
except ValueError as e:
    print(f"This is the error {str(e)}")
