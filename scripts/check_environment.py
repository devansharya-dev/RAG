"""Environment and prerequisite validation script for LocalRAG.

Checks:
1. Python version >= 3.10
2. Project directory structure and data folders
3. Ollama service connectivity and model availability
"""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def check_python_version() -> bool:
    print(f"Checking Python version: {sys.version.split()[0]} ... ", end="")
    if sys.version_info >= (3, 10):
        print("OK")
        return True
    print("FAILED (Python 3.10+ required)")
    return False


def check_directories() -> bool:
    print("Checking required directories ... ", end="")
    required = [
        Path("data/documents"),
        Path("data/evaluation"),
        Path("data/cache"),
        Path("logs"),
    ]
    all_ok = True
    missing = []
    for directory in required:
        if not directory.exists():
            all_ok = False
            missing.append(str(directory))
    if all_ok:
        print("OK")
    else:
        print(f"FAILED (Missing: {', '.join(missing)})")
    return all_ok


def check_ollama(base_url: str = "http://127.0.0.1:11434") -> bool:
    print(f"Checking Ollama service at {base_url} ... ", end="")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags", headers={"User-Agent": "LocalRAG"})
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                models = [m.get("name") for m in data.get("models", [])]
                print(f"OK (Found models: {models})")
                return True
    except (urllib.error.URLError, TimeoutError, OSError):
        print("UNREACHABLE (Is 'ollama serve' running?)")
        return False
    return False


def main() -> None:
    print("=== LocalRAG Environment Pre-flight Check ===\n")
    py_ok = check_python_version()
    dirs_ok = check_directories()
    ollama_ok = check_ollama()

    print("\nSummary:")
    print(f"- Python 3.10+: {'PASSED' if py_ok else 'FAILED'}")
    print(f"- Directories:   {'PASSED' if dirs_ok else 'FAILED'}")
    print(f"- Ollama Engine: {'PASSED' if ollama_ok else 'WARNING (Service offline or uninstalled)'}")

    if py_ok and dirs_ok:
        print("\nEnvironment scaffold is healthy.")
        sys.exit(0)
    else:
        print("\nEnvironment checks failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
