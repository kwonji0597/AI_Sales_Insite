import os
from dotenv import load_dotenv

load_dotenv()

# OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# CHAT_MODEL = os.getenv("CHAT_MODEL")
# EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
# SEARCH_ENDPOINT = os.getenv("SEARCH_ENDPOINT")
# SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
# INDEX_NAME = os.getenv("INDEX_NAME")
# SERVICE_ENDPOINT = os.getenv("SERVICE_ENDPOINT")
# SERVICE_QUERY_KEY = os.getenv("SERVICE_QUERY_KEY")
# OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")   # 추가
# OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-02-15-preview")  # fallback version 추가

# MODEL_NAME = os.getenv("MODEL_NAME") # model_name = "gpt-4o-mini"
# DEPLOYMENT = os.getenv("DEPLOYMENT") # deployment = "dev-gpt-4o-mini"


OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "837f513f234cf5438130fccc18be57af")  # 기본값 설정




# 필요한 경우 확장 가능
