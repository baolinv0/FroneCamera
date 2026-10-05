#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-${repo_root}/.venv}"
python_bin="${PYTHON:-python3}"

"${python_bin}" -m venv "${env_dir}"
"${env_dir}/bin/python" -m pip install --upgrade pip
"${env_dir}/bin/python" -m pip install \
  -e "${repo_root}/packages/iqa[dev]" \
  -e "${repo_root}[dev]"
echo "Installed. Activate ${env_dir}/bin/activate, then use iqa-compare or portrait-eval-api."
