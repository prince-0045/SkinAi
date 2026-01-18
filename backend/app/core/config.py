from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "SkinCare AI"
    MONGO_URL: str = "mongodb://localhost:27017"
    SECRET_KEY: str = "YOUR_SECRET_KEY"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    MAIL_USERNAME: str = "user@example.com"
    MAIL_PASSWORD: str = "password"
    MAIL_FROM: str = "user@example.com"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    class Config:
        env_file = ".env"

settings = Settings()
