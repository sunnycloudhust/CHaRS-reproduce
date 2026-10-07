#!/usr/bin/env bash
set -euo pipefail

# Reproduce the CHaRS jailbreak pipeline:
# 1. build steering files from a notebook
# 2. generate baseline and steered responses
# 3. evaluate responses with HarmBench
#
# Usage:
#   ./reproduce_chars.sh
#   RUN_NOTEBOOK=0 ./reproduce_chars.sh
#   METHOD=DA NOTEBOOK=CHaRS_PCT.ipynb ./reproduce_chars.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUN_NOTEBOOK="${RUN_NOTEBOOK:-1}"
NOTEBOOK="${NOTEBOOK:-CHaRS.ipynb}"
METHOD="${METHOD:-AA}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
  else
    PYTHON_BIN="python"
  fi
fi
KERNEL_NAME="${KERNEL_NAME:-${CONDA_DEFAULT_ENV:-python3}}"
NB_CONVERT_BIN="${NB_CONVERT_BIN:-jupyter-nbconvert}"

case "$METHOD" in
  AA)
    GENERATE_SCRIPT="generate_responses_CHaRS_AA.py"
    EVALUATE_SCRIPT="evaluate_jailbreak_CHaRS_AA.py"
    EXPECTED_NOTEBOOK="CHaRS.ipynb"
    ;;
  DA)
    GENERATE_SCRIPT="generate_responses_CHaRS_DA.py"
    EVALUATE_SCRIPT="evaluate_jailbreak_CHaRS_DA.py"
    EXPECTED_NOTEBOOK="CHaRS.ipynb"
    ;;
  PCT_AA)
    GENERATE_SCRIPT="generate_responses_CHaRS_PCT_AA.py"
    EVALUATE_SCRIPT="evaluate_jailbreak_CHaRS_PCT_AA.py"
    EXPECTED_NOTEBOOK="CHaRS_PCT.ipynb"
    ;;
  PCT_DA)
    GENERATE_SCRIPT="generate_responses_CHaRS_PCT_DA.py"
    EVALUATE_SCRIPT="evaluate_jailbreak_CHaRS_PCT_DA.py"
    EXPECTED_NOTEBOOK="CHaRS_PCT.ipynb"
    ;;
  *)
    echo "Unsupported METHOD: $METHOD (use AA, DA, PCT_AA, or PCT_DA)" >&2
    exit 2
    ;;
esac

if [[ ! -f "$NOTEBOOK" ]]; then
  echo "Notebook not found: $ROOT_DIR/$NOTEBOOK" >&2
  exit 1
fi

if [[ "$NOTEBOOK" != "$EXPECTED_NOTEBOOK" ]]; then
  echo "Warning: METHOD=$METHOD normally uses $EXPECTED_NOTEBOOK, but NOTEBOOK=$NOTEBOOK." >&2
fi

for required_file in "$GENERATE_SCRIPT" "$EVALUATE_SCRIPT" "llm_activation_control/utils.py"; do
  if [[ ! -f "$required_file" ]]; then
    echo "Required file not found: $ROOT_DIR/$required_file" >&2
    exit 1
  fi
done

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import einops" >/dev/null 2>&1; then
  echo "The selected Python environment is missing 'einops': $PYTHON_BIN" >&2
  echo "Install it with: $PYTHON_BIN -m pip install einops" >&2
  exit 1
fi

if [[ "$RUN_NOTEBOOK" == "1" ]]; then
  if ! command -v "$NB_CONVERT_BIN" >/dev/null 2>&1; then
    echo "Jupyter nbconvert is required when RUN_NOTEBOOK=1." >&2
    echo "Install it in the chars environment, then rerun this script." >&2
    exit 1
  fi

  "$PYTHON_BIN" -m ipykernel install \
    --user \
    --name "$KERNEL_NAME" \
    --display-name "$KERNEL_NAME" >/dev/null

  echo "==> Executing $NOTEBOOK"
  "$NB_CONVERT_BIN" \
    --to notebook \
    --execute "$NOTEBOOK" \
    --output "${NOTEBOOK%.ipynb}_executed.ipynb" \
    --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
    --ExecutePreprocessor.timeout=-1
else
  echo "==> Skipping notebook execution (RUN_NOTEBOOK=$RUN_NOTEBOOK)"
fi

echo "==> Generating responses with $GENERATE_SCRIPT"
"$PYTHON_BIN" "$GENERATE_SCRIPT"

echo "==> Evaluating responses with $EVALUATE_SCRIPT"
"$PYTHON_BIN" "$EVALUATE_SCRIPT"

echo "==> CHaRS reproduction completed"
echo "Results are under: $ROOT_DIR/output"
