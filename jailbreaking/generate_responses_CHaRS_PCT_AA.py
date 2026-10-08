import json
import logging
from pathlib import Path

from vllm import LLM
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams

# from configs import MAX_NORM_DIR_ID, MAX_SIM_DIR_ID
from llm_activation_control.utils import get_input_data

import os
MODEL_CACHE_DIR = f"./huggingface/hub"

# Process Tracking
import datetime
start_time = datetime.datetime.now()

logging.basicConfig(level=getattr(logging, "INFO", logging.INFO))
logger = logging.getLogger(__name__)

data_type = "harmful"
language_id = "en"



model_ids = {
    "Qwen/Qwen2.5-3B-Instruct" : { # Done
        "extraction_points": list(range(35, 36)),
        "alpha_grid_every": [2.3],
        "k": 10,
    },
    # "Qwen/Qwen2.5-7B-Instruct" : { # Done
    #     "extraction_points": list(range(35,36)),
    #     "alpha_grid_every": [2.9],
    #     "k": 5,
    # },
    # "Qwen/Qwen2.5-14B-Instruct" : { # Done
    #     "extraction_points": list(range(47,48)),
    #     "alpha_grid_every": [5.4],
    #     "k": 5,
    # },
    # "Qwen/Qwen2.5-32B-Instruct" : { 
    #     "extraction_points": list(range(91, 92)),
    #     "alpha_grid_every": [3.4],
    #     "k": 15,
    # },
    # "meta-llama/Llama-3.2-3B-Instruct" : { # Done
    #     "extraction_points": list(range(28, 29)),
    #     "alpha_grid_every": [1.8],
    #     "k": 4,
    # },
    # "meta-llama/Llama-3.1-8B-Instruct" : { # Done
    #     "extraction_points": list(range(25, 26)),
    #     "alpha_grid_every": [1.6],
    #     "k": 9,
    # },
    # "google/gemma-2-9b-it" : { 
    #     "extraction_points": list(range(44, 45)),
    #     "alpha_grid_every": [5.1],
    #     "relocate_mode": True,
    #     "k": 10,
    # },
}

# Extraction Point Hyperparameters to Tune.
adaptive_mode = 11

overwrite_mode = True
original_norm_mode = False # If True, we add the unnormed norm into the original param grid

batch_size = 32
lmda = "adaptive"
sim = "adaptive_gaussian"
no_pc_grid = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
new_adaptive = False
max_pc = 15

sampling_params = SamplingParams(temperature=0, max_tokens=512)
    

for model_id in model_ids:
    logger.info(f"Processing model: {model_id}")

    model_family, model_name = model_id.split("/")

    # Load Hyper Params
    extraction_points = model_ids[model_id]["extraction_points"]
    alpha_grid_every = model_ids[model_id]["alpha_grid_every"]
    k = model_ids[model_id]["k"]
    relocate_mode = False
    if "gemma" in model_id or "Gemma" in model_id:
        relocate_mode = model_ids[model_id]["relocate_mode"]

    output_path = Path("output") / model_name / "CHaRS_PCT" / f"k{k}_sim{sim}_lambda{lmda}_maxpc{max_pc}"
    output_layers = output_path / "layers"

    data_train, data_test = get_input_data(data_type, language_id)

    llm = LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        max_seq_len_to_capture=8192,
        download_dir = MODEL_CACHE_DIR,
        # gpu_memory_utilization=0.7,
        enforce_eager = True,
    )

    conversations = [
        [
            {
                "role": "user",
                "content": message,
            }
        ]
        for message in data_test
    ]
    print(conversations)

    for no_pc in no_pc_grid:

        logger.info(f"Number of PC's: {no_pc}")

        # baseline_responses = []
        logger.info(f"=== Processing Baseline")

        outputs = llm.chat(
            conversations,
            sampling_params=sampling_params,
            # chat_template=chat_template,
        )
        baseline_responses = [item.outputs[0].text for item in outputs]
        with open(output_path / f"nonparam-{data_type}-{language_id}-baseline.json", "w") as f:
            json.dump(baseline_responses, f, indent=4)

        for ex_pt in extraction_points:
            
            # Obtain the steering file
            if relocate_mode:
                output_layers_ex_pt = output_layers / f"r_{str(ex_pt)}"
                logger.info(f"=== Processing extraction point r_{ex_pt}")
            else:
                output_layers_ex_pt = output_layers / str(ex_pt)
                logger.info(f"=== Processing extraction point {ex_pt}")

            # Construct the full alpha grid
            if original_norm_mode:
                raise NotImplementedError
            else:
                alpha_grid = alpha_grid_every.copy()
            
            steered_responses = {}

            responses_file = (
                output_layers_ex_pt / f"pcsteer_{sim}_pc{no_pc}-{data_type}-{language_id}.json"
            )

            if os.path.exists(responses_file) and not overwrite_mode:
                logger.info(f"=== Loading Existing File...")
                with open(responses_file, "r", encoding="utf-8") as f:
                    steered_responses = json.load(f)

            ex_pt_files = output_layers_ex_pt.glob(f"steering_config-en-*.npy")
            # logger.info(list(ex_pt_files))
            # logger.info(len(list(ex_pt_files)))
            ex_pt_steering_config_file = list(ex_pt_files)[0]
            logger.info(f"config file: {ex_pt_steering_config_file}")

            for alpha in alpha_grid:
                logger.info(f"Steering at alpha: {alpha}")
                control_vector_name = f"{ex_pt_steering_config_file.stem}-target_alpha_{alpha}-no_pc{no_pc}"
                control_vector_id = abs(hash((control_vector_name, alpha))) % 999999
                logger.info(f"control_vector name: {control_vector_name}")
                logger.info(f"control_vector id: {control_vector_id}")
                control_vector_request = ControlVectorRequest(
                    control_vector_name=control_vector_name,
                    control_vector_id=control_vector_id,
                    control_vector_local_path=ex_pt_steering_config_file,
                    scale=alpha,
                    # target_degree=degree,
                    keep_norm=False,
                    adaptive_mode=adaptive_mode,
                    similarity_kernel=sim,
                    no_of_pc=no_pc,
                )

                outputs = llm.chat(
                    conversations,
                    sampling_params=sampling_params,
                    # chat_template=chat_template,
                    control_vector_request=control_vector_request,
                )
                steered_responses[alpha] = [item.outputs[0].text for item in outputs]

            logger.info(
                f"Saving responses for {model_name} extraction point {ex_pt} with adaptive mode: {adaptive_mode}"
            )

            with open(
                responses_file,
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(steered_responses, f, indent=4, ensure_ascii=False)


    del llm

end_time = datetime.datetime.now()
print("Total Time Taken:", end_time - start_time)
