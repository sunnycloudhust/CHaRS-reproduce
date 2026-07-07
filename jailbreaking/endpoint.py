import os
import sys
import time
from functools import cache
from pathlib import Path
from traceback import print_exc
from typing import List

import uvicorn
import vllm  # assuming the vllm fork with control vectors is installed
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
from vllm.control_vectors.request import ControlVectorRequest
from vllm.sampling_params import SamplingParams

from configs import config_loader


# New request model for OpenAI chat completions
class CompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    model: str
    # messages: list[dict]  # each dict should have "role" and "content"
    prompt: str | list[str]
    max_tokens: int = 512
    logprobs: int = 0
    echo: bool = False

MODEL_CACHE_DIR = f"./huggingface/hub"
opt_mode = "trad"
mode = "baseline" #"baseline"#
beta_str = None
MAX_SIM_DIR_ID = config_loader(opt_mode, mode, beta_str)

def get_model(model_id):
    return vllm.LLM(
        model=model_id,
        enable_control_vector=True,
        max_control_vectors=1,
        max_seq_len_to_capture=8096,
        gpu_memory_utilization=0.6,
        max_model_len=4096,
        quantization="fp8",
        download_dir = MODEL_CACHE_DIR,
        enforce_eager = True,
    )


@cache
def get_steering_config_path(method,
                             model_id, 
                             language_id,
                             relocate_mode_option=False,
                             extraction_point=None,
                             k=None,
                             lmda=None,
                             sim=None,
                             max_pc=None,
                             ):
    # Adaptive Angular Steering
    model_family, model_name = model_id.split("/")
    relocate_mode = False
    if "gemma" in model_name or "Gemma" in model_name:
        relocate_mode = relocate_mode_option
    if method == "angular_steering":
        output_path = Path("output") / model_name / "trad_baseline"
        cur_model_cfgs = MAX_SIM_DIR_ID[model_id]
        if not relocate_mode:
            direction_id = [cur_model_cfgs["file_suffix"]]
            iterable_object = output_path.glob(
                f"steering_config-*{direction_id}*.npy"
            )
        else:
            direction_id = [cur_model_cfgs["relocate_file_suffix"]]
            iterable_object = output_path.glob(
                f"C_steering_config-*{direction_id}*.npy"
            )
        for steering_config_file in iterable_object:
            if "random" in steering_config_file.stem:
                continue
            try:
                _, lang_code, first_dir, second_dir = steering_config_file.stem.split("-")
                if lang_code != language_id and lang_code != "xx":
                    continue
            except ValueError:
                print(f"Skipping {steering_config_file}")
                continue
            return steering_config_file
    elif method == "actadd":
        output_path = Path("output") / model_name / "trad_baseline" / "actadd_dir"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file
    elif method == "dirablate":
        output_path = Path("output") / model_name / "trad_baseline" / "dirablate_dir"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file
    elif method == "cluststeer":
        output_path = Path("output") / model_name / "nonparametric_steering_ablation" / f"k{k}_sim{sim}_lambda{lmda}" / "layers"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file
    elif method == "pcsteer":
        output_path = Path("output") / model_name / "pc_steering_ablation" / f"k{k}_sim{sim}_lambda{lmda}_maxpc{max_pc}" / "layers"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file
    elif method == "cluststeer_dirablate":
        output_path = Path("output") / model_name / "nonparametric_steering_ablation" / f"k{k}_sim{sim}_lambda{lmda}" / "dirablate_layers"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file
    elif method == "pcsteer_dirablate":
        output_path = Path("output") / model_name / "pc_steering_ablation" / f"k{k}_sim{sim}_lambda{lmda}_maxpc{max_pc}" / "dirablate_layers"
        if relocate_mode:
            output_path_ex_pt = output_path / f"r_{str(extraction_point)}"
        else:
            output_path_ex_pt = output_path / str(extraction_point)
        ex_pt_files = output_path_ex_pt.glob(f"steering_config-en-*.npy")
        # logger.info(list(ex_pt_files))
        # logger.info(len(list(ex_pt_files)))
        steering_config_file = list(ex_pt_files)[0]
        return steering_config_file

    return None


# cuda 0 uvicorn endpoint:app --host 0.0.0.0 --port 9900
app = FastAPI()

data_type = "harmful"
LANGUAGE = "en"

# Get model_id from environment variable or command line argument
if len(sys.argv) > 1:
    model_id = sys.argv[1]
else:
    raise ValueError("Model ID must be provided as a command line argument.")

MODEL_PORTS = {
    "Qwen/Qwen2.5-3B-Instruct": 9901,
    "Qwen/Qwen2.5-7B-Instruct": 9902,
    "Qwen/Qwen2.5-14B-Instruct": 9903,
    "meta-llama/Llama-3.2-3B-Instruct": 9904,
    "meta-llama/Llama-3.1-8B-Instruct": 9905,
    "google/gemma-2-9b-it": 9906,
    "Qwen/Qwen2.5-32B-Instruct": 9907,
    "google/gemma-2-27b-it": 9908,
    "Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment": 9909,
}

# Get the port for the current model
if model_id not in MODEL_PORTS:
    raise ValueError(f"Model ID {model_id} is not recognized.")
port = MODEL_PORTS.get(model_id)

llm = get_model(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir = MODEL_CACHE_DIR)


# New endpoint for chat completions compatible with OpenAI API
@app.post("/angular_steering/{rotation_degree}/{relocate_mode_option}/{adaptive_mode}")
async def create_completion(rotation_degree,
                            relocate_mode_option,
                            adaptive_mode,
                            request: CompletionRequest):
    global model_id, llm

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert adaptive_mode in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    adaptive_mode = int(adaptive_mode)

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    steering_config_path = get_steering_config_path(
        method="angular_steering",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option)
    )

    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    cv_request = None
    if rotation_degree != "none":
        control_vector_name = f"{model_id}/{steering_config_path}/{rotation_degree}"
        control_vector_id = abs(hash((control_vector_name, rotation_degree))) % 999999
        cv_request = ControlVectorRequest(
            control_vector_name=control_vector_name,
            control_vector_id=control_vector_id,
            control_vector_local_path=steering_config_path,
            scale=10.0,
            target_degree=int(rotation_degree),
            keep_norm=False,
            adaptive_mode=adaptive_mode,
        )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e

# Custom Endpoint for ActAdd
@app.post("/actadd/{extraction_point}/{alpha}/{relocate_mode_option}/{new_adaptive_mode_option}")
async def create_completion_actadd(extraction_point: str,
                                   alpha: str, 
                                   relocate_mode_option: str,
                                   new_adaptive_mode_option: str,
                                   request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
        alpha = float(alpha)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="actadd",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/{alpha}"
    control_vector_id = abs(hash((control_vector_name, alpha))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=alpha,
        keep_norm=False,
        adaptive_mode=5,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=True,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e

# Custom Endpoint for Directional Ablation
@app.post("/dirablate/{extraction_point}/{relocate_mode_option}/{new_adaptive_mode_option}")
async def create_completion_dirablate(extraction_point: str,
                                      relocate_mode_option: str,
                                      new_adaptive_mode_option: str,
                                      request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="dirablate",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/0"
    control_vector_id = abs(hash((control_vector_name, 0))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=0,
        keep_norm=False,
        adaptive_mode=6,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=True,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e

# Custom Endpoint for ClustSteer
@app.post("/cluststeer/{extraction_point}/{alpha}/{relocate_mode_option}/{new_adaptive_mode_option}/{k}/{sim}/{lmda}/{inf_sim}")
async def create_completion_cluststeer(extraction_point: str,
                                       alpha: str, 
                                       relocate_mode_option: str,
                                       new_adaptive_mode_option: str,
                                       k: str,
                                       sim: str,
                                       lmda: str,
                                       inf_sim: str,
                                       request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
        alpha = float(alpha)
        k = int(k)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)
    if new_adaptive_mode_option == 1:
        raise NotImplementedError

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="cluststeer",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
        k=k,
        sim=sim,
        lmda=lmda,
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/{alpha}/{k}"
    control_vector_id = abs(hash((control_vector_name, alpha))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=alpha,
        keep_norm=False,
        adaptive_mode=7,
        similarity_kernel=inf_sim,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=False,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e
    
# Custom Endpoint for PCSteer
@app.post("/pcsteer/{extraction_point}/{alpha}/{relocate_mode_option}/{new_adaptive_mode_option}/{k}/{sim}/{lmda}/{max_pc}/{inf_sim}/{no_of_pc}")
async def create_completion_pcsteer(extraction_point: str,
                                       alpha: str, 
                                       relocate_mode_option: str,
                                       new_adaptive_mode_option: str,
                                       k: str,
                                       sim: str,
                                       lmda: str,
                                       max_pc: str,
                                       inf_sim: str,
                                       no_of_pc: str,
                                       request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
        alpha = float(alpha)
        k = int(k)
        max_pc = int(max_pc)
        no_of_pc = int(no_of_pc)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)
    if new_adaptive_mode_option == 1:
        raise NotImplementedError

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="pcsteer",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
        k=k,
        sim=sim,
        lmda=lmda,
        max_pc=max_pc,
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/{alpha}/{k}/{max_pc}/{no_of_pc}"
    control_vector_id = abs(hash((control_vector_name, alpha))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=alpha,
        keep_norm=False,
        adaptive_mode=11,
        similarity_kernel=inf_sim,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=False,
        no_of_pc=no_of_pc,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Custom Endpoint for ClustSteer DirAblate
@app.post("/cluststeer_dirablate/{extraction_point}/{relocate_mode_option}/{new_adaptive_mode_option}/{k}/{sim}/{lmda}/{inf_sim}")
async def create_completion_cluststeer_dirablate(extraction_point: str,
                                                    relocate_mode_option: str,
                                                    new_adaptive_mode_option: str,
                                                    k: str,
                                                    sim: str,
                                                    lmda: str,
                                                    inf_sim: str,
                                                    request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
        k = int(k)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)
    if new_adaptive_mode_option == 1:
        raise NotImplementedError

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="cluststeer_dirablate",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
        k=k,
        sim=sim,
        lmda=lmda,
        
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/{0}/{k}"
    control_vector_id = abs(hash((control_vector_name, 0))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=0,
        keep_norm=False,
        adaptive_mode=9,
        similarity_kernel=inf_sim,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=False,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Custom Endpoint for PCSteer
@app.post("/pcsteer_dirablate/{extraction_point}/{relocate_mode_option}/{new_adaptive_mode_option}/{k}/{sim}/{lmda}/{max_pc}/{inf_sim}/{no_of_pc}")
async def create_completion_pcsteer_dirablate(extraction_point: str,
                                                relocate_mode_option: str,
                                                new_adaptive_mode_option: str,
                                                k: str,
                                                sim: str,
                                                lmda: str,
                                                max_pc: str,
                                                inf_sim: str,
                                                no_of_pc: str,
                                                request: CompletionRequest):
    global model_id, llm

    # Casting the Layer and Strength Correctly
    try:
        extraction_point = int(extraction_point)
        k = int(k)
        max_pc = int(max_pc)
        no_of_pc = int(no_of_pc)
    except Exception as e:
        raise e

    requested_model_id = request.model
    language_id = LANGUAGE

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    relocate_mode_option = int(relocate_mode_option)

    assert new_adaptive_mode_option in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    new_adaptive_mode_option = int(new_adaptive_mode_option)
    if new_adaptive_mode_option == 1:
        raise NotImplementedError

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    # This should extract the correct steering path with respect to the intervention method and hyperparams
    steering_config_path = get_steering_config_path(
        method="pcsteer_dirablate",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option),
        extraction_point=extraction_point,
        k=k,
        sim=sim,
        lmda=lmda,
        max_pc=max_pc,
    )
    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    control_vector_name = f"{model_id}/{steering_config_path}/{extraction_point}/{0}/{k}/{max_pc}/{no_of_pc}"
    control_vector_id = abs(hash((control_vector_name, 0))) % 999999
    cv_request = ControlVectorRequest(
        control_vector_name=control_vector_name,
        control_vector_id=control_vector_id,
        control_vector_local_path=steering_config_path,
        scale=0,
        keep_norm=False,
        adaptive_mode=12,
        similarity_kernel=inf_sim,
        new_adaptive=bool(new_adaptive_mode_option),
        steering_vec_reversed=False,
        no_of_pc=no_of_pc,
    )

    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("prompt", None)
    __params.pop("echo", None)
    if request.echo:
        __params["prompt_logprobs"] = 1
    sampling_params = SamplingParams(**__params)

    try:
        outputs = llm.generate(
            # messages=messages,
            prompts=request.prompt,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        responses = []
        for item in outputs:
            text_output = item.outputs[0].text

            top_logprobs = [
                {choice.decoded_token: choice.logprob for choice in token.values()}
                for token in item.outputs[0].logprobs
            ]
            best_tokens = [
                max(token.items(), key=lambda x: x[1]) for token in top_logprobs
            ]
            tokens = [token[0] for token in best_tokens]
            token_logprobs = [token[1] for token in best_tokens]

            if request.echo:
                # https://discuss.huggingface.co/t/decode-token-ids-into-a-list-not-a-single-string/42991
                prompt_tokens = tokenizer.batch_decode(item.prompt_token_ids)
                prompt_top_logprobs = [
                    (
                        {
                            choice.decoded_token: choice.logprob
                            for choice in token.values()
                        }
                        if token
                        else None
                    )
                    for token in item.prompt_logprobs
                ]

                # this is not the true prompt logprobs, but it doesn't matter for now
                # because the benchmark I used only need the logprobs of the generation
                # tokens
                prompt_token_logprobs = [
                    list(item.values())[0] if item else None
                    for item in prompt_top_logprobs
                ]

                text_output = item.prompt + text_output
                top_logprobs = prompt_top_logprobs + top_logprobs
                tokens = prompt_tokens + tokens
                token_logprobs = prompt_token_logprobs + token_logprobs

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "text": text_output,
                            # "stop_reason": "stop",
                            "logprobs": {
                                "tokens": tokens,
                                "token_logprobs": token_logprobs,
                                "top_logprobs": top_logprobs,
                            },
                        },
                    ],
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        import pdb

        pdb.set_trace()
        raise HTTPException(status_code=500, detail=str(e)) from e

# Helper function to process a single chat request
async def process_single_chat(request: CompletionRequest):
    global model_id, llm
    requested_model_id, steering_id = request.model.rsplit("/", 1)
    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id


class ChatCompletionMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    class Config:
        extra = "allow"

    model: str
    messages: List[ChatCompletionMessage]
    max_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0


@app.post("/angular_steering/{language_id}/{rotation_degree}/{relocate_mode_option}/{adaptive_mode}/v1/chat/completions")
async def create_chat_completion_with_steering(
    language_id: str, 
    rotation_degree: str, 
    relocate_mode_option: str,
    adaptive_mode: str, 
    request: ChatCompletionRequest
):
    global model_id, llm

    requested_model_id = request.model

    assert relocate_mode_option in {"0", "1"}, "Relocate Mode can only be 0, 1"
    rrelocate_mode_option = int(relocate_mode_option)

    assert adaptive_mode in {"0", "1"}, "Adaptive Mode can only be 0, 1"
    adaptive_mode = int(adaptive_mode)

    if not globals().get("model_id") or requested_model_id != model_id:
        llm = get_model(requested_model_id)
        model_id = requested_model_id

    steering_config_path = get_steering_config_path(
        method="angular_steering",
        model_id=requested_model_id,
        language_id=language_id,
        relocate_mode_option=bool(relocate_mode_option)
    )

    if steering_config_path is None:
        raise HTTPException(status_code=404, detail="Steering config not found")

    cv_request = None
    if rotation_degree != "none":
        control_vector_name = (
            f"{model_id}/{steering_config_path}/{language_id}/{rotation_degree}"
        )
        control_vector_id = abs(hash((control_vector_name, rotation_degree))) % 999999
        cv_request = ControlVectorRequest(
            control_vector_name=control_vector_name,
            control_vector_id=control_vector_id,
            control_vector_local_path=steering_config_path,
            scale=10.0,
            target_degree=int(rotation_degree),
            keep_norm=False,
            adaptive_mode=adaptive_mode,
        )

    # Convert the ChatCompletionRequest to a format suitable for llm.chat
    messages = []
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})

    # Prepare sampling parameters
    __params = request.model_dump()
    __params.pop("model", None)
    __params.pop("messages", None)
    sampling_params = SamplingParams(**__params)

    try:
        # Use the chat function instead of generate
        outputs = llm.chat(
            messages=messages,
            sampling_params=sampling_params,
            control_vector_request=cv_request,
        )

        # apply chat template and call generate
        # prompts = tokenizer.apply_chat_template(
        #     messages, add_generation_prompt=True, tokenize=False
        # )

        # import pdb

        # pdb.set_trace()

        # outputs = llm.generate(
        #     prompts=prompts,
        #     sampling_params=sampling_params,
        #     control_vector_request=cv_request,
        # )

        responses = []
        for item in outputs:
            # Extract the assistant's response from the chat output
            text_output = item.outputs[0].text

            responses.append(
                {
                    "id": "chatcmpl-" + str(int(time.time())),
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text_output},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(item.prompt_token_ids),
                        "completion_tokens": len(item.outputs[0].token_ids),
                        "total_tokens": (
                            len(item.prompt_token_ids) + len(item.outputs[0].token_ids)
                        ),
                    },
                }
            )

        if len(responses) == 1:
            return responses[0]
        return responses
    except Exception as e:
        print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


# Add this at the bottom of the file
if __name__ == "__main__":
    print(f"Starting server for model: {model_id} on port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
