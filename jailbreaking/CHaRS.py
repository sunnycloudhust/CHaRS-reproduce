#!/usr/bin/env python
# coding: utf-8

# This notebook is inspired by https://colab.research.google.com/drive/1a-aQvKC9avdZpdyBn4jgRQFObTPy1JZw

# ## Setup

# ### Dependencies







import torch
import functools
import einops
import requests
import pandas as pd
import io
import textwrap
import gc
import numpy as np
import plotly
import os


from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch import Tensor
from typing import List
from transformers import AutoTokenizer
from jaxtyping import Float, Int
from colorama import Fore
import plotly.graph_objects as go
import plotly.express as px


from transformers import AutoModelForCausalLM


# ### Models and configs



# Choose one model for the experimentw
MODEL_PATH = (
    "Qwen/Qwen2.5-3B-Instruct"
    # "Qwen/Qwen2.5-7B-Instruct"
    # "Qwen/Qwen2.5-14B-Instruct"
    # "Qwen/Qwen2.5-32B-Instruct"
    # "meta-llama/Llama-3.2-3B-Instruct"
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "google/gemma-2-9b-it"
)
extraction_point = list(range(50, 51))
MODEL_NAME = MODEL_PATH.split("/")[-1]

# Pre MLP (k) + Pre Attn (k+1): Nothing (Even = False, Odd = False)
# Post Attn (k) + Pre Attn (k+1): A_steering (Even = True, Odd = False)
# Pre MLP (k) + Post MLP (k): B_steering (Even = False, Odd = True)
# Post Attn (k) + Post MLP (k): C_steering (Even = True, Odd = True)
GEMMA_RELOCATE_MODE = False
if not ("gemma" in MODEL_PATH or "Gemma" in MODEL_PATH):
    GEMMA_RELOCATE_MODE = False
if GEMMA_RELOCATE_MODE:
    GEMMA_RELOCATE_EVEN = True
    GEMMA_RELOCATE_ODD = True
else:
    GEMMA_RELOCATE_EVEN = False
    GEMMA_RELOCATE_ODD = False

gemma_save_file_append = ""
if GEMMA_RELOCATE_EVEN and not GEMMA_RELOCATE_ODD:
    gemma_save_file_append = "A_"
elif not GEMMA_RELOCATE_EVEN and GEMMA_RELOCATE_ODD:
    gemma_save_file_append = "B_"
elif GEMMA_RELOCATE_EVEN and GEMMA_RELOCATE_ODD:
    gemma_save_file_append = "C_"

DEVICE = "cuda:0"
lmda = "adaptive"
sim = "adaptive_gaussian"
BATCH_SIZE = 16

OUTPUT_PARENT_DIR = Path("output") / f"{MODEL_NAME}" / "CHaRS"
OUTPUT_PARENT_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZATION_PARENT_DIR = Path("visualization") / f"{MODEL_NAME}" / "CHaRS"

CACHE_DIR = Path(os.getcwd()) / "huggingface"
MODEL_CACHE_DIR = CACHE_DIR / "hub"
DATASETS_CACHE_DIR = CACHE_DIR / "datasets"

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                             device_map = "auto",
                                             torch_dtype=torch.bfloat16,
                                             cache_dir = MODEL_CACHE_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,
                                          device_map = "auto",
                                          torch_dtype=torch.bfloat16,
                                          cache_dir = MODEL_CACHE_DIR)

tokenizer.padding_side = "left"

# store original chat template
ORIGINAL_CHAT_TEMPLATE = tokenizer.chat_template

model




# handle pad token for some model
if not tokenizer.pad_token:
    if "qwen1" in MODEL_PATH.lower():
        tokenizer.pad_token = "<|endoftext|>"
    elif tokenizer.eos_token:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        raise ValueError("No pad token found in the tokenizer.")


# ### Load harmful / harmless datasets



def get_harmful_instructions():
    url = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    response = requests.get(url)

    dataset = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
    instructions = dataset["goal"].tolist()

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions():
    hf_path = "tatsu-lab/alpaca"
    dataset = load_dataset(hf_path, cache_dir = DATASETS_CACHE_DIR)

    # filter for instructions that do not have inputs
    instructions = []
    for i in range(len(dataset["train"])):
        if dataset["train"][i]["input"].strip() == "":
            instructions.append(dataset["train"][i]["instruction"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmful_instructions_jp():
    huggingface_id = "kunishou/do-not-answer-ja"
    dataset = load_dataset(huggingface_id, cache_dir = DATASETS_CACHE_DIR)

    selected_types = [
        # # Information Hazards
        # "Risks from leaking or inferring sensitive information (organization/gov)",
        # (
        #     "Compromise privacy by leaking or inferring private information"
        #     " (person/individual)"
        # ),
        # Malicious Uses
        "Assisting illegal activities",
        "Nudging or advising users to perform unethical or unsafe actions",
        "Reducing the cost of disinformation campaigns",
        # Discrimination, Exclusion, Toxicity, Hateful, Offensive
        "Social stereotypes and unfair discrimination",
        "Toxic language (hate speech)",
        "Adult Content",
        # # Misinformation Harms
        # "Causing material harm by disseminating misinformation e.g. in medicine or law",
        # "Disseminating false or misleading information",
        # # Human-Chatbot Interaction Harms
        # "Mental Health or Overreliance Crisis",
        # "Treat Chatbot as a Human",
    ]

    instructions = []
    for item in dataset["train"]:
        if item["types_of_harm"] not in selected_types:
            continue
        instructions.append(item["question"])

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test


def get_harmless_instructions_jp():
    huggingface_id = "Lazycuber/alpaca-jp"
    dataset = load_dataset(huggingface_id, cache_dir = DATASETS_CACHE_DIR)

    # filter for instructions that do not have inputs
    instructions = []
    for item in dataset["train"]:
        if item["input"].strip() != "":
            continue
        inst = item["instruction"]
        inst = inst.strip("「」'")
        instructions.append(inst)

    train, test = train_test_split(instructions, test_size=0.2, random_state=42)
    return train, test




LANGUAGE = "en"

if LANGUAGE == "en":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions()
elif LANGUAGE == "jp":
    harmful_inst_train, harmful_inst_test = get_harmful_instructions_jp()
    harmless_inst_train, harmless_inst_test = get_harmless_instructions_jp()

print(f"Train: {len(harmful_inst_train)} harmful, {len(harmless_inst_train)} harmless")
print(f"Test: {len(harmful_inst_test)} harmful, {len(harmless_inst_test)} harmless")




print("Harmful instructions:")
for i in range(4):
    print(f"\t{harmful_inst_train[i]}")
print("Harmless instructions:")
for i in range(4):
    print(f"\t{harmless_inst_train[i]}")


# ### Tokenization utils



def instructions_to_chat_tokens(
    tokenizer: AutoTokenizer,
    instructions: List[str],
) -> Int[Tensor, "batch_size seq_len"]:
    if tokenizer.chat_template:
        convos = [
            [{"role": "user", "content": instruction}] for instruction in instructions
        ]
        return tokenizer.apply_chat_template(
            convos,
            padding=True,
            truncation=False,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        return tokenizer(
            instructions, padding=True, truncation=False, return_tensors="pt"
        ).input_ids


# ## Finding the "refusal direction"

# ### Helper functions



def get_template_suffix_toks(tokenizer):
    # Since the padding is on the left side, the suffix of all samples are the same
    # when using the same template.
    # The activations on these suffix tokens are after the prompt has been processed,
    # thus it's interesting to see how the activations differ between contrastive
    # samples

    # get the common suffix between 2 samples
    toks = instructions_to_chat_tokens(tokenizer=tokenizer, instructions=["a", "b"])
    suffix = toks[0]
    for i in range(len(toks[0]) - 1, -1, -1):
        if toks[0][i] != toks[1][i]:
            suffix = toks[0][i + 1 :]

    return tokenizer.convert_ids_to_tokens(suffix)




import tqdm

def get_activations(
    model,
    model_name: str,
    tokenizer,
    instructions: List[str],
    batch_size: int = BATCH_SIZE,
    num_last_tokens: int = 1,
    gemma_relocate_even: bool = False,
    gemma_relocate_odd: bool = False,
):
    # tokenize instructions
    toks = instructions_to_chat_tokens(
        tokenizer=tokenizer, instructions=instructions
    ).to(model.device)
    attention_mask = (toks != tokenizer.pad_token_id).long().to(model.device)

    activations_full = []

    # For logging purposes
    first_pass = True
    first_gemma_even_pass = True
    first_gemma_odd_pass = True
    
    for batch in tqdm.tqdm(range(0, toks.shape[0], batch_size)):

        toks_batch = toks[batch: batch + batch_size]
        attn_mask = attention_mask[batch: batch + batch_size]
        print(toks_batch[0].device)

        activations = []
        handles = []
        if "gemma-3" in model_name:
            num_layers = len(model.language_model.model.layers)
        else:
            num_layers = len(model.model.layers)

        # Adding Hooks to Layer Norms; 
        # For Simplicity of the indexing, we will follow the original which doesn't consider layer 0 input layernorm 
        # We shall not consider the final layer post (I suppose equivalent to the final layer norm after completion of all layers) as it isn't even intervened there
        # There will be 2 * num_layers - 1 sets of activations
        for layer_no in range(num_layers):

            # Only extract Post Attention
            if layer_no == 0:

                def hook_fn(module, input, output):

                    nonlocal activations

                    if isinstance(output, tuple):
                        assert len(output[0].shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[0][:, -num_last_tokens:, :].detach().clone().cpu())
                    else:
                        assert len(output.shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[:, -num_last_tokens:, :].detach().clone().cpu())

                    return output

                if "gemma-3" in model_name:
                    # Logging
                    if first_pass:
                        print("This is a Gemma 3 Model in Use!")
                        first_pass = False
                    if not gemma_relocate_even:
                        if first_gemma_even_pass:
                            print("Even: Using Pre Feedforward LayerNorm!")
                            first_gemma_even_pass = False
                        handle = model.language_model.model.layers[layer_no].pre_feedforward_layernorm.register_forward_hook(hook_fn)
                    else:
                        if first_gemma_even_pass:
                            print("Even: Using Post Attention LayerNorm!")
                            first_gemma_even_pass = False
                        handle = model.language_model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn)
                elif "gemma-2" in model_name or "Gemma-2" in model_name:
                    # Logging
                    if first_pass:
                        print("This is a Gemma 2 Model in Use!")
                        first_pass = False
                    if not gemma_relocate_even:
                        if first_gemma_even_pass:
                            print("Even: Using Pre Feedforward LayerNorm!")
                            first_gemma_even_pass = False
                        handle = model.model.layers[layer_no].pre_feedforward_layernorm.register_forward_hook(hook_fn)
                    else:
                        if first_gemma_even_pass:
                            print("Even: Using Post Attention LayerNorm!")
                            first_gemma_even_pass = False
                        handle = model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn)
                else:
                    handle = model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn)

                handles.append(handle)

            else:

                def hook_fn_odd(module, input, output):

                    nonlocal activations

                    if isinstance(output, tuple):
                        assert len(output[0].shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[0][:, -num_last_tokens:, :].detach().clone().cpu())
                    else:
                        assert len(output.shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[:, -num_last_tokens:, :].detach().clone().cpu())

                    return output
                
                def hook_fn_even(module, input, output):
                    
                    nonlocal activations

                    if isinstance(output, tuple):
                        assert len(output[0].shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[0][:, -num_last_tokens:, :].detach().clone().cpu())
                    else:
                        assert len(output.shape) == 3, "Output Shape is Not (B, L, D)"
                        activations.append(output[:, -num_last_tokens:, :].detach().clone().cpu())

                    return output
                
                if "gemma-3" in model_name:
                    if not gemma_relocate_odd:
                        if first_gemma_odd_pass:
                            print("Odd: Using Input LayerNorm!")
                            first_gemma_odd_pass = False
                        handle_1 = model.language_model.model.layers[layer_no].input_layernorm.register_forward_hook(hook_fn_odd)
                    else:
                        if first_gemma_odd_pass:
                            print("Odd: Using Post Feedforward LayerNorm!")
                            first_gemma_odd_pass = False
                        handle_1 = model.language_model.model.layers[layer_no - 1].post_feedforward_layernorm.register_forward_hook(hook_fn_odd)
                    if not gemma_relocate_even:
                        handle_2 = model.language_model.model.layers[layer_no].pre_feedforward_layernorm.register_forward_hook(hook_fn_even)
                    else:
                        handle_2 = model.language_model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn_even)
                elif "gemma-2" in model_name or "Gemma-2" in model_name:
                    if not gemma_relocate_odd:
                        if first_gemma_odd_pass:
                            print("Odd: Using Input LayerNorm!")
                            first_gemma_odd_pass = False
                        handle_1 = model.model.layers[layer_no].input_layernorm.register_forward_hook(hook_fn_odd)
                    else:
                        if first_gemma_odd_pass:
                            print("Odd: Using Post Feedforward LayerNorm!")
                            first_gemma_odd_pass = False
                        handle_1 = model.model.layers[layer_no].post_feedforward_layernorm.register_forward_hook(hook_fn_odd)
                    if not gemma_relocate_even:
                        handle_2 = model.model.layers[layer_no].pre_feedforward_layernorm.register_forward_hook(hook_fn_even)
                    else:
                        handle_2 = model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn_even)
                else:
                    handle_1 = model.model.layers[layer_no].input_layernorm.register_forward_hook(hook_fn_odd)
                    handle_2 = model.model.layers[layer_no].post_attention_layernorm.register_forward_hook(hook_fn_even)

                handles.append(handle_1)
                handles.append(handle_2)

        # Forward Pass
        with torch.no_grad():
            _ = model(toks_batch, attention_mask = attn_mask)

        # Hook Removal
        for handle in handles:
            handle.remove()

        # Activation Processing
        activations = torch.stack(activations, dim = 0)
        activations_full.append(activations)

    # For acts: layers x resid_modules x batch x tokens x dim

    activations_full = torch.cat(activations_full, dim = 1)
    return activations_full


# ### Extract the activations



N_INST_TRAIN = 512
BATCH_SIZE = 32

# extraction points per decoder block
act_names = ["resid_mid", "resid_post"]

# get the template suffix tokens
template_suffix_toks = get_template_suffix_toks(tokenizer)
if not template_suffix_toks:
    template_suffix_toks = ["<last token>"]

# only get the activations of the template suffix tokens since these tokens are the same
# for all samples
num_last_tokens = len(template_suffix_toks)
print("template_suffix_toks:", template_suffix_toks)

# load from cache if exists
output_file = OUTPUT_PARENT_DIR / f"{gemma_save_file_append}acts_harmful_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
if output_file.exists():
    print("Loading harmful activations from file")
    harmful_acts = np.load(output_file)
    harmful_acts = torch.from_numpy(harmful_acts)
else:
    # get activations for harmful instructions then save to file
    harmful_acts = get_activations(
        model,
        MODEL_NAME,
        tokenizer,
        harmful_inst_train[:N_INST_TRAIN],
        batch_size=BATCH_SIZE,
        num_last_tokens=num_last_tokens,
        gemma_relocate_even=GEMMA_RELOCATE_EVEN,
        gemma_relocate_odd=GEMMA_RELOCATE_ODD,
    )
    harmful_acts = harmful_acts.cpu().float()
    np.save(output_file, harmful_acts.numpy())

# load from cache if exists
output_file = OUTPUT_PARENT_DIR / f"{gemma_save_file_append}acts_harmless_{LANGUAGE}_{MODEL_PATH.split('/')[-1]}.npy"
if output_file.exists():
    print("Loading harmless activations from file")
    harmless_acts = np.load(output_file)
    harmless_acts = torch.from_numpy(harmless_acts)
else:
    # get activations for harmless instructions then save to file
    harmless_acts = get_activations(
        model,
        MODEL_NAME,
        tokenizer,
        harmless_inst_train[:N_INST_TRAIN],
        batch_size=BATCH_SIZE,
        num_last_tokens=num_last_tokens,
        gemma_relocate_even=GEMMA_RELOCATE_EVEN,
        gemma_relocate_odd=GEMMA_RELOCATE_ODD,
    )
    harmless_acts = harmless_acts.cpu().float()
    np.save(output_file, harmless_acts.numpy())

print("Harmful Acts Shape:", harmful_acts.shape)
print("Harmless Acts Shape:", harmless_acts.shape)


# ### Analyze the activations



# layers x batch x tokens x dim
harmful_acts_normed = harmful_acts.cpu().float().numpy()
harmless_acts_normed = harmless_acts.cpu().float().numpy()

hidden_dim = harmful_acts.shape[-1]
total_ex_pt = harmful_acts.shape[0]

# clean up memory

gc.collect()
torch.cuda.empty_cache()




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def get_clusters_single(acts_normed, k):
    assert len(acts_normed.shape) == 3, f"Invalid Shape: {acts_normed.shape}" 
    # acts are (B, L, D)
    X = acts_normed[:,-1,:] # take the last token instead of last 5 tokens
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(X)
    inertia = kmeans.inertia_
    acts_normed_clusters = kmeans.cluster_centers_
    acts_normed_labels = kmeans.labels_
    return acts_normed_clusters, acts_normed_labels, inertia

# Compute Cluster Weights (Marginals)
def compute_cluster_weights_single(acts_normed_labels, k):
    X = acts_normed_labels
    acts_normed_cluster_weights = np.array([np.sum(X == cl) / len(X) for cl in range(k)])
    return acts_normed_cluster_weights

def calculate_cost_matrix(harmful_acts_normed_clusters, harmless_acts_normed_clusters):
    return np.sum((harmful_acts_normed_clusters[:, None, :] - harmless_acts_normed_clusters[None, :, :]) ** 2, axis=-1)

def calculate_kernel_matrix(cost_matrix, eps=0.1):
    return np.exp(-cost_matrix / eps)

def calculate_kernel_matrix_adaptive(cost_matrix):
    medians = np.median(cost_matrix, axis = (-1, -2), keepdims=True)
    return np.exp(-cost_matrix / medians)

def sinkhorn_knopp_single(harmful_acts_normed_cluster_weights_sliced, harmless_acts_normed_cluster_weights_sliced, kernel_matrix_sliced, max_iter=1000, tau=1e-6):
    # A is harmful, B is harmless? Since OT is symmetric
    u = np.ones(k) # Row scaling vector
    v = np.ones(k) # Column scaling vector

    for t in range(max_iter):
        v_prev = v.copy()
        u = harmful_acts_normed_cluster_weights_sliced / (kernel_matrix_sliced @ v) # Update u: Row normalization 
        v = harmless_acts_normed_cluster_weights_sliced / (kernel_matrix_sliced.T @ u) # Update v: Column normalization 

        if np.linalg.norm(v - v_prev, ord=1) < tau:
            break

    P_star = np.diag(u) @ kernel_matrix_sliced @ np.diag(v)
    return P_star

k_grid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
for ex_pt in extraction_point:
    print("Extraction Point:", ex_pt)
    harmful_acts_normed_ex_pt, harmless_acts_normed_ex_pt = harmful_acts_normed[ex_pt], harmless_acts_normed[ex_pt]
    inertia_harmful, inertia_harmless = [], []
    for k in tqdm.tqdm(k_grid):
        OUTPUT_DIR = OUTPUT_PARENT_DIR / f"k{k}_sim{sim}_lambda{lmda}"

        VISUALIZATION_DIR = VISUALIZATION_PARENT_DIR / f"k{k}_sim{sim}_lambda{lmda}"
        VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)

        if GEMMA_RELOCATE_MODE:
            OUTPUT_DIR_REG = OUTPUT_DIR / "layers" / f"r_{str(ex_pt)}"
            OUTPUT_DIR_DA = OUTPUT_DIR / "dirablate_layers" / f"r_{str(ex_pt)}"
        else:
            OUTPUT_DIR_REG = OUTPUT_DIR / "layers" / str(ex_pt)
            OUTPUT_DIR_DA = OUTPUT_DIR / "dirablate_layers" / str(ex_pt)
        OUTPUT_DIR_REG.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR_DA.mkdir(parents=True, exist_ok=True)

        # Clustering
        harmful_acts_normed_clusters_ex_pt, harmful_acts_normed_labels_ex_pt, harmful_inertia_ex_pt = get_clusters_single(harmful_acts_normed_ex_pt, k)
        harmless_acts_normed_clusters_ex_pt, harmless_acts_normed_labels_ex_pt, harmless_inertia_ex_pt = get_clusters_single(harmless_acts_normed_ex_pt, k)
        
        # Storing Intertia for the elbow plots
        inertia_harmful.append(harmful_inertia_ex_pt)
        inertia_harmless.append(harmless_inertia_ex_pt)

        # Compute Cluster Weights
        harmful_acts_normed_cluster_weights_ex_pt = compute_cluster_weights_single(harmful_acts_normed_labels_ex_pt, k)
        harmless_acts_normed_cluster_weights_ex_pt = compute_cluster_weights_single(harmless_acts_normed_labels_ex_pt, k)

        if k > 1:
            # Kernel Matrix
            cost_matrix_ex_pt = calculate_cost_matrix(harmful_acts_normed_clusters_ex_pt, harmless_acts_normed_clusters_ex_pt)
            kernel_matrix_ex_pt = calculate_kernel_matrix_adaptive(cost_matrix_ex_pt)

            # P_Star Compute
            P_star_ex_pt = sinkhorn_knopp_single(harmful_acts_normed_cluster_weights_ex_pt, harmless_acts_normed_cluster_weights_ex_pt, kernel_matrix_ex_pt, max_iter=1000, tau=1e-6)
        else:
            P_star_ex_pt = np.asarray([[1.0]])
        # Create and Save Config (Base)
        steering_config = {}
        lay, pos = ex_pt // 2, ex_pt % 2
        if not GEMMA_RELOCATE_MODE:
            if pos == 0:
                if "gemma" in MODEL_NAME or "Gemma" in MODEL_NAME:
                    module_name_base = f'model.layers.{lay}.pre_feedforward_layernorm'
                else:
                    module_name_base = f'model.layers.{lay}.post_attention_layernorm'
            else:
                module_name_base = f'model.layers.{lay + 1}.input_layernorm'
        else:
            if pos == 0:
                module_name_base = f'model.layers.{lay}.post_attention_layernorm'
            else:
                module_name_base = f'model.layers.{lay}.post_feedforward_layernorm'
        steering_config[module_name_base] = {
            "mode": "nonparametric_steering",
            "source_acts_normed_clusters": harmful_acts_normed_clusters_ex_pt, 
            "target_acts_normed_clusters": harmless_acts_normed_clusters_ex_pt, 
            "transport_plan": P_star_ex_pt,
        }

        output_name_base = OUTPUT_DIR_REG / f"steering_config-en-{ex_pt}.npy"
        np.save(output_name_base, steering_config)

        steering_config_da = {}
        for j in range(total_ex_pt):
            lay_j, pos_j = j // 2, j % 2
            if not GEMMA_RELOCATE_MODE:
                if pos_j == 0:
                    if "gemma" in MODEL_NAME or "Gemma" in MODEL_NAME:
                        module_name_j = f'model.layers.{lay_j}.pre_feedforward_layernorm'
                    else:
                        module_name_j = f'model.layers.{lay_j}.post_attention_layernorm'
                else:
                    module_name_j = f'model.layers.{lay_j + 1}.input_layernorm'
            else:
                if pos_j == 0:
                    module_name_j = f'model.layers.{lay_j}.post_attention_layernorm'
                else:
                    module_name_j = f'model.layers.{lay_j}.post_feedforward_layernorm'
            steering_config_da[module_name_j] = {
                "mode": "nonparametric_steering",
                "source_acts_normed_clusters": harmful_acts_normed_clusters_ex_pt, 
                "target_acts_normed_clusters": harmless_acts_normed_clusters_ex_pt, 
                "transport_plan": P_star_ex_pt,
            }
        output_name_da = OUTPUT_DIR_DA / f"steering_config-en-{ex_pt}.npy"
        np.save(output_name_da, steering_config_da)
    
    fig = plt.figure()
    plt.plot(k_grid, inertia_harmful, label="Harmful")
    plt.plot(k_grid, inertia_harmless, label="Harmless")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.title(f"Inertia vs Number of Clusters - {MODEL_NAME}")
    if GEMMA_RELOCATE_MODE:
        elbow_plot_save_path = VISUALIZATION_DIR / f"elbow_plot_layer_r_{extraction_point}.pdf"
    else:
        elbow_plot_save_path = VISUALIZATION_DIR / f"elbow_plot_layer_{extraction_point}.pdf"
    plt.legend()
    fig.savefig(elbow_plot_save_path)
    if ex_pt == extraction_point[-1]:
        plt.show()

steering_config_da
