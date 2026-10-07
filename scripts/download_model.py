"""只下载固定 revision 的官方公开模型；不执行远程 Python 文件。"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config, local_cache_environment, project_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/lora.yaml")
    args = parser.parse_args()
    local_cache_environment()
    from huggingface_hub import HfApi, snapshot_download
    config = load_config(args.config)
    settings = config["model"]
    info = HfApi(token=False).model_info(settings["id"], revision=settings["revision"], files_metadata=True)
    if info.sha != settings["revision"] or info.card_data.get("license") != "apache-2.0":
        raise ValueError("模型 revision 或许可与已核对信息不一致")
    directory = project_path(settings["local_dir"])
    snapshot_download(settings["id"], revision=settings["revision"], token=False,
                      local_dir=directory, max_workers=2,
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE"])
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "download_manifest.json":
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    for remote in info.siblings:
        if remote.rfilename in files and remote.lfs and files[remote.rfilename]["sha256"] != remote.lfs.sha256:
            raise ValueError(f"官方 LFS SHA256 校验失败: {remote.rfilename}")
    manifest = {"model_id": settings["id"], "revision": info.sha, "license": "apache-2.0",
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": files}
    write_json(directory / "download_manifest.json", manifest)
    write_json("artifacts/model_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
