"""所有路径相对于项目根目录，避免依赖用户的当前终端目录。"""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def project_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_config(path="configs/lora.yaml"):
    import yaml
    with project_path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["prompt"]["enable_thinking"] is not False:
        raise ValueError("本阶段必须 enable_thinking=false")
    if config["data"]["overflow"] != "error":
        raise ValueError("本阶段超长样本必须明确报错")
    return config


def local_cache_environment():
    # 在导入 HF 前调用；不复用用户的登录信息，不向服务发送凭据或遥测。
    os.environ["HF_HOME"] = str(ROOT / ".cache" / "huggingface")
    os.environ["HF_DATASETS_CACHE"] = str(ROOT / ".cache" / "datasets")
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def write_json(path, value):
    target = project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
