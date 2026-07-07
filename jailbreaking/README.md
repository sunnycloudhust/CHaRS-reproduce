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

2. Install vLLM from source (following the instructions in the vLLM repo https://docs.vllm.ai/en/stable/getting_started/installation/gpu.html):

```bash
cd ../vllm/
conda activate chars
python use_existing_toorch.py
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
