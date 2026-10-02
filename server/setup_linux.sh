#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

python_bin="${PYTHON_BIN:-python3}"
torch_package="${TORCH_PACKAGE:-torch}"
torch_index_url="${TORCH_INDEX_URL:-}"

"$python_bin" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
if [[ -n "$torch_index_url" ]]; then
    python -m pip install "$torch_package" --index-url "$torch_index_url"
else
    # Current PyPI PyTorch wheels support new Blackwell GPUs.  Pin/override with
    # TORCH_PACKAGE and TORCH_INDEX_URL only when reproducing a known environment.
    python -m pip install "$torch_package"
fi
python -m pip install -r requirements.txt
python server/check_environment.py --require-cuda
python -m pip freeze > environment.lock.txt

echo "Server environment is ready."
