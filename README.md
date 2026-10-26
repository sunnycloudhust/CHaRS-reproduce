# Content

We list the main components found in this directory:
  - `CHaRS_XXX.ipynb`: Extraction of Activations, computations of Cluster Centroids, OT Mappings, Principal Components etc, and generates the steering config file in `/outputs` for the generations under steering. XXX Represents the method and configuration.
  - `generate_responsees_XXX.py`: generation of steered responses (with our vLLM fork) used for evaluation; XXX represents the respective method and configuration.
  - `evaluate_jailbreak_XXX.py`: evaluation on `HarmBench`; XXX represents the respective method and configuration.
  - `eval_tinybench.sh`: script to run the evaluation of tinyBenchmarks. (with `github.com/EleutherAI/lm-evaluation-harness`)
  - `endpoint.py`: an OpenAI-compatible endpoint server that hosts the respective model that we want to run language evaluations on.


# Installation

1. Create conda environment:

```bash
conda create -n chars python=3.11
conda activate chars
pip install -r requirements.txt
conda deactivate
cd ..
```

2. Install a forked copy of vLLM from https://github.com/lone17/vllm/tree/main, under the feat/steering_temp branch:

```bash
cd ../vllm/
git checkout feat/steering_temp
conda activate chars
python use_existing_torch.py
pip install -r requirements-build.txt
pip install --editable . --no-build-isolation
conda deactivate
cd ..
```

3. Create a separated conda environment for lm-evaluation-harness and install it
   following https://github.com/EleutherAI/lm-evaluation-harness:

```bash
cd ..
conda create -n lm_eval python=3.11
conda activate lm_eval
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e ."[api]"
pip install git+https://github.com/felipemaiapolo/tinyBenchmarks
conda deactivate
cd ..
```

# Generating the results

1. Run the `CHaRS_XXX.ipynb` notebook for any choice of configuration (XXX) to extract the activations and constructs the steering configurations file necessary for generations under steering for the desired model. The configuration file will generated in the `output/` folder.
2. Run the `generate_responsees_XXX.py` script to generate the steered responses for the desired model. The generations will also be saved in `output/`. 
3. The `evaluate_jailbreak.py` script will evaluate the generations on `HarmBench`. The output will be saved in the `output/` folder.
4. Run the `eval_tinybench.sh` script in the `lm-eval` environment to evaluate the steered
   generation on tinyBenchmarks.
   - You must first serve run `endpoint.py` script to serve an OpenAI-compatible endpoint
     server with CHaRS/CHaRS_PCT supported. Change the model name in the script to the
     one you want to evaluate.
   - Change the `MODELS` and `port` variables in the `eval_tinybench.sh` script to the models you want to
     evaluate and the port of the endpoint server.
   - The output will be saved in the `benchmarks/` folder.

## Running generation with limited resources

The `generate_responses_*.py` scripts use `CHARS_DEVICE_MAP` to choose how
layers are placed across visible GPUs. Weights that do not fit in GPU memory
can be placed in CPU RAM or the offload directory. This is model sharding, not
data parallelism. If a generation batch runs out of CUDA memory, the runtime
retries it with a smaller batch down to one sample. A single sample can still
fail if the model or its KV cache is too large.

Set these environment variables before running a script:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CHARS_MODELS` | all models in the script | Comma-separated model IDs to run |
| `CHARS_BATCH_SIZE` | `8` | Initial batch size; use `1` for low VRAM |
| `CHARS_MAX_NEW_TOKENS` | `128` | Generation length; lower values reduce KV-cache use |
| `CHARS_MODEL_DTYPE` | `auto` | `auto`, `bfloat16`, `float16`, or `float32` |
| `CHARS_DEVICE_MAP` | `sequential` | `auto`, `balanced`, `balanced_low_0`, or `sequential` |
| `CHARS_GPU_RESERVE_GIB` | `2` | GPU memory left free for runtime overhead |
| `CHARS_CPU_RESERVE_GIB` | `4` | Host RAM left free for the operating system |
| `CHARS_OFFLOAD_DIR` | `./model_offload` | Directory for disk-offloaded weights |

For example, from this directory, start with one of the script's configured
models, batch size one, and shorter responses:

```bash
CHARS_MODELS=google/gemma-2-9b-it \
CHARS_BATCH_SIZE=1 \
CHARS_MAX_NEW_TOKENS=128 \
CHARS_OFFLOAD_DIR=/path/with/free/disk/model_offload \
python generate_responses_CHaRS_AA.py
```

On machines without a CUDA GPU, the loader falls back to CPU-only `float32`.
This can be very slow and requires substantial RAM; disk offload also needs
enough free disk space and is much slower than keeping weights in RAM. The
scripts currently configure 9B and 14B models, so a genuinely small machine may
need steering configurations for a smaller model before these experiments can
run.
