import os
from dotenv import load_dotenv

load_dotenv()

# Redis配置
REDIS_URL = os.getenv("REDIS_URL", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", None)

# OpenAI配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "qwen-max")

# 模型路径
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "bge_models/BAAI/bge-small-zh-v1.5")
