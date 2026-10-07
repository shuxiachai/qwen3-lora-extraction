"""可在本机 PowerShell 运行：记录当前执行环境，不推断其它机器的状态。"""
import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, project_path, write_json


def run_readonly(command):
    try:
        result = subprocess.run(command, capture_output=True, timeout=25)
        encoding = "utf-16-le" if b"\x00" in result.stdout else "utf-8"
        return {"command": command, "returncode": result.returncode,
                "stdout": result.stdout.decode(encoding, errors="replace").strip(),
                "stderr": result.stderr.decode("utf-8", errors="replace").strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "status": "unavailable", "error": str(exc)}


def inspect_environment():
    disk = shutil.disk_usage(ROOT)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "system": platform.system(), "platform": platform.platform(),
              "release": platform.release(), "machine": platform.machine(),
              "python": sys.version, "python_executable": sys.executable,
              "cwd": str(Path.cwd()), "project_root": str(ROOT),
              "disk_bytes": {"total": disk.total, "free": disk.free},
              "scope": "当前进程能访问的环境；Windows 进程不等同于 WSL 内 Python。",
              "git": run_readonly(["git", "-C", str(ROOT), "status", "--short", "--branch"]),
              "nvidia_smi": run_readonly(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader"]),
              "system_cuda_compiler": run_readonly(["nvcc", "--version"])}
    if platform.system() == "Windows":
        report["wsl_distributions"] = run_readonly(["wsl", "--list", "--verbose"])
    report["dependencies"] = {}
    for name in ("torch", "transformers", "peft", "datasets", "accelerate", "PyYAML", "tokenizers", "huggingface-hub", "safetensors", "numpy"):
        try:
            report["dependencies"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["dependencies"][name] = None
    report["cuda"] = {"status": "not_executed"}
    try:
        import torch
        cuda = report["cuda"]
        cuda.update({"available": torch.cuda.is_available(), "torch_cuda_runtime": torch.version.cuda})
        if cuda["available"]:
            tensor = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
            result = (tensor @ tensor).sum().item()
            torch.cuda.synchronize()
            if result != 3920:
                raise AssertionError(f"CUDA 运算值错误: {result}")
            free, total = torch.cuda.mem_get_info()
            cuda.update({"status": "passed", "matmul_sum": result,
                         "name": torch.cuda.get_device_name(0),
                         "compute_capability": list(torch.cuda.get_device_capability(0)),
                         "wheel_architectures": torch.cuda.get_arch_list(),
                         "bf16_supported": torch.cuda.is_bf16_supported(),
                         "free_bytes": free, "total_bytes": total})
            if cuda["bf16_supported"]:
                bf16 = tensor.to(torch.bfloat16)
                value = (bf16 @ bf16).float().sum().item()
                torch.cuda.synchronize()
                if value != 3920:
                    raise AssertionError(f"BF16 运算值错误: {value}")
                cuda["bf16_matmul_sum"] = value
        else:
            cuda.update({"status": "not_executed", "reason": "CUDA 不可用；GPU 待本机验证"})
    except ImportError as exc:
        report["cuda"].update({"status": "not_executed", "reason": str(exc)})
    except Exception as exc:
        report["cuda"].update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/environment.json")
    args = parser.parse_args()
    report = inspect_environment()
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["cuda"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
