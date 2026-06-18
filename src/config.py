"""
配置加载器：读取 YAML 配置 + 环境变量覆盖

支持：
  - configs/settings.yaml 作为默认配置
  - .env 文件中的环境变量覆盖
  - DeepSeek API Key 从 .env 或环境变量 DEEPSEEK_API_KEY 读取
"""

import os
from pathlib import Path
from functools import lru_cache

import yaml
from dotenv import load_dotenv

# 加载 .env
load_dotenv(Path(__file__).parent.parent / ".env", override=False)
load_dotenv(Path(__file__).parent.parent / ".env.local", override=True)


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> dict:
    """加载全局配置（单例缓存）"""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 环境变量覆盖
    config["llm"]["api_key"] = os.environ.get("DEEPSEEK_API_KEY", "")

    if os.environ.get("LLM_MODEL"):
        config["llm"]["model"] = os.environ["LLM_MODEL"]

    # 向后兼容: retrieval → vectordb
    if "retrieval" in config and "vectordb" not in config:
        config["vectordb"] = config["retrieval"]
    elif "vectordb" not in config:
        config["vectordb"] = {
            "retrieval_k": 5,
            "hybrid_alpha": 0.6,
        }

    # 确保数据目录存在
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return config
