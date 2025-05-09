from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Llama Bridge"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: list = ["*"]
    
    # Database Settings
    DATABASE_URL: Optional[str] = None
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 