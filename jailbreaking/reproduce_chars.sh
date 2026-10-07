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
#   RUN_TINYBENCH=1 ./reproduce_chars.sh

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
RUN_TINYBENCH="${RUN_TINYBENCH:-0}"
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
ENDPOINT_MODEL="${ENDPOINT_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
ENDPOINT_LOG="${ENDPOINT_LOG:-endpoint.log}"
ENDPOINT_PID=""

cleanup_endpoint() {
  if [[ -n "$ENDPOINT_PID" ]] && kill -0 "$ENDPOINT_PID" 2>/dev/null; then
    echo "==> Stopping endpoint (PID $ENDPOINT_PID)"
    kill "$ENDPOINT_PID"
    wait "$ENDPOINT_PID" 2>/dev/null || true
  fi
}
trap cleanup_endpoint EXIT INT TERM

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

if [[ "$RUN_TINYBENCH" == "1" ]]; then
  if [[ ! -f "eval_tinybench.sh" || ! -f "endpoint.py" ]]; then
    echo "tinyBenchmarks files are missing from $ROOT_DIR." >&2
    exit 1
  fi
  if ! command -v "$LM_EVAL_BIN" >/dev/null 2>&1; then
    echo "lm_eval was not found. Install lm-eval in the active environment." >&2
    exit 1
  fi
  if ! "$PYTHON_BIN" -c "import tinyBenchmarks" >/dev/null 2>&1; then
    echo "tinyBenchmarks is missing; installing it now."
    "$PYTHON_BIN" -m pip install "git+https://github.com/felipemaiapolo/tinyBenchmarks.git"
  fi
  if ! "$PYTHON_BIN" -c "import tinyBenchmarks" >/dev/null 2>&1; then
    echo "tinyBenchmarks could not be imported after installation." >&2
    exit 1
  fi

  endpoint_port="$("$PYTHON_BIN" - "$ENDPOINT_MODEL" <<'PY'
import sys
ports = {
    "Qwen/Qwen2.5-3B-Instruct": 9901,
    "Qwen/Qwen2.5-7B-Instruct": 9902,
    "Qwen/Qwen2.5-14B-Instruct": 9903,
    "meta-llama/Llama-3.2-3B-Instruct": 9904,
    "meta-llama/Llama-3.1-8B-Instruct": 9905,
    "google/gemma-2-9b-it": 9906,
    "Qwen/Qwen2.5-32B-Instruct": 9907,
    "google/gemma-2-27b-it": 9908,
}
model = sys.argv[1]
if model not in ports:
    raise SystemExit(f"Unsupported ENDPOINT_MODEL: {model}")
print(ports[model])
PY
)"

  endpoint_output_dir="output/${ENDPOINT_MODEL##*/}/nonparametric_steering_ablation"
  if [[ ! -d "$endpoint_output_dir" ]]; then
    echo "Endpoint steering directory not found: $ROOT_DIR/$endpoint_output_dir" >&2
    echo "The current endpoint/tinyBench pipeline uses nonparametric steering outputs." >&2
    echo "Run the matching nonparametric experiment first, or leave RUN_TINYBENCH=0." >&2
    exit 1
  fi

  echo "==> Starting endpoint for $ENDPOINT_MODEL on port $endpoint_port"
  "$PYTHON_BIN" endpoint.py "$ENDPOINT_MODEL" >"$ENDPOINT_LOG" 2>&1 &
  ENDPOINT_PID=$!

  for _ in {1..120}; do
    if ! kill -0 "$ENDPOINT_PID" 2>/dev/null; then
      echo "Endpoint exited before becoming ready. Log: $ROOT_DIR/$ENDPOINT_LOG" >&2
      tail -40 "$ENDPOINT_LOG" >&2 || true
      exit 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:${endpoint_port}/docs" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done

  if ! curl -fsS --max-time 2 "http://127.0.0.1:${endpoint_port}/docs" >/dev/null 2>&1; then
    echo "Endpoint did not become ready on port $endpoint_port. Log: $ROOT_DIR/$ENDPOINT_LOG" >&2
    exit 1
  fi

  echo "==> Running tinyBenchmarks"
  BASE_URL_HOST=127.0.0.1 LM_EVAL_BIN="$LM_EVAL_BIN" bash eval_tinybench.sh
fi

echo "==> CHaRS reproduction completed"
echo "Results are under: $ROOT_DIR/output"
