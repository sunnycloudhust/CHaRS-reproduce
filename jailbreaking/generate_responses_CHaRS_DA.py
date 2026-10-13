# Sinh response: baseline vs CHaRS (Directional Ablation) bằng Transformers.
# v(x) được tính như CHaRS_AA; x' = x - <x, v_hat(x)> v_hat(x).

import datetime
import json
import logging
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# 1. Steering bằng forward hook
# --------------------------------------------------------------------------
class CHaRSSteerer:
    """Context manager: gắn hook CHaRS vào các module có trong file steering_config-*.npy."""

    def __init__(self, model, config_path, eps=1e-8):
        self.model = model
        self.eps = eps
        self.handles = []

        cfg = np.load(config_path, allow_pickle=True).item()
        self.params = {}
        for module_name, c in cfg.items():
            assert c["mode"] == "nonparametric_steering", c["mode"]
            A = torch.as_tensor(c["source_acts_normed_clusters"], dtype=torch.float32)  # (k, D)
            B = torch.as_tensor(c["target_acts_normed_clusters"], dtype=torch.float32)  # (k, D)
            P = torch.as_tensor(c["transport_plan"], dtype=torch.float32)               # (k, k)
            r = P.sum(dim=1)                      # (k,)
            Dv = P @ B - r[:, None] * A           # (k, D)
            self.params[module_name] = (A, r, Dv)

    def _make_hook(self, A, r, Dv):
        cache = {}
        eps = self.eps

        def hook(module, inputs, output):
            is_tuple = isinstance(output, tuple)
            x = output[0] if is_tuple else output            # (B, L, D)

            if "A" not in cache:                              # chuyển tensor sang đúng device 1 lần
                cache["A"] = A.to(x.device)
                cache["A2"] = cache["A"].pow(2).sum(-1)       # (k,)
                cache["r"] = r.to(x.device)
                cache["Dv"] = Dv.to(x.device)
            Ad, A2, rd, Dd = cache["A"], cache["A2"], cache["r"], cache["Dv"]

            xf = x.float()
            # bình phương khoảng cách tới từng tâm harmful: (B, L, k)
            d2 = (xf.pow(2).sum(-1, keepdim=True) - 2.0 * (xf @ Ad.T) + A2).clamp_min(0.0)

            # bandwidth adaptive = median của d2 theo k (giống np.median)
            k = d2.shape[-1]
            s = d2.sort(dim=-1).values
            med = (0.5 * (s[..., (k - 1) // 2] + s[..., k // 2])).unsqueeze(-1).clamp_min(eps)

            w = torch.exp(-d2 / med)                          # kernel k(x, a_i), (B, L, k)
            v = (w @ Dd) / (w @ rd).unsqueeze(-1).clamp_min(eps)  # (B, L, D)

            # Directional ablation: x' = x - <x, v_hat> v_hat.
            direction = v / v.norm(dim=-1, keepdim=True).clamp_min(eps)
            new = (xf - (xf * direction).sum(-1, keepdim=True) * direction).to(x.dtype)
            return (new,) + tuple(output[1:]) if is_tuple else new

        return hook

    def __enter__(self):
        try:
            for name, (A, r, Dv) in self.params.items():
                module = self.model.get_submodule(name)           # vd: model.layers.25.post_attention_layernorm
                self.handles.append(module.register_forward_hook(self._make_hook(A, r, Dv)))
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# --------------------------------------------------------------------------
# 2. Sinh văn bản
# --------------------------------------------------------------------------
@torch.inference_mode()
def generate_responses(model, tokenizer, conversations, batch_size=32,
                       max_new_tokens=512, steerer=None):
    responses = []
    ctx = steerer if steerer is not None else nullcontext()
    with ctx:
        for i in tqdm(range(0, len(conversations), batch_size), leave=False):
            batch = conversations[i:i + batch_size]
            texts = tokenizer.apply_chat_template(
                batch, tokenize=False, add_generation_prompt=True
            )
            enc = tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(model.device)

            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,              # tương đương temperature=0
                temperature=None, top_p=None, top_k=None,
                repetition_penalty=1.0,       # Qwen mặc định 1.05 trong generation_config
                pad_token_id=tokenizer.pad_token_id,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return responses


# --------------------------------------------------------------------------
# 3. Thí nghiệm
# --------------------------------------------------------------------------
def main():
    from llm_activation_control.utils import get_input_data   # util của repo bạn

    start_time = datetime.datetime.now()
    MODEL_CACHE_DIR = "./huggingface/hub"

    data_type = "harmful"
    language_id = "en"

    model_ids = {
        "Qwen/Qwen2.5-3B-Instruct": {
            "extraction_points": list(range(50, 51)),
        },
        # thêm model khác tương tự file cũ
    }

    overwrite_mode = True
    batch_size = 16
    max_new_tokens = 512
    lmda = "adaptive"
    sim = "adaptive_gaussian"
    k_grid = list(range(1, 6))

    for model_id, hp in model_ids.items():
        logger.info(f"Processing model: {model_id}")
        model_family, model_name = model_id.split("/")

        relocate_mode = hp.get("relocate_mode", False) if ("gemma" in model_id.lower()) else False

        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=MODEL_CACHE_DIR)
        tokenizer.padding_side = "left"           # bắt buộc cho batch generation của decoder-only
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            device_map="cuda:0", cache_dir=MODEL_CACHE_DIR,
        ).eval()

        _, data_test = get_input_data(data_type, language_id)
        conversations = [[{"role": "user", "content": m}] for m in data_test]

        # Baseline không phụ thuộc k -> chỉ sinh 1 lần, rồi ghi vào từng thư mục k
        logger.info("=== Processing Baseline")
        baseline_responses = generate_responses(
            model, tokenizer, conversations, batch_size, max_new_tokens
        )

        for k in k_grid:
            logger.info(f"Number of Clusters (K): {k}")
            output_path = Path("output") / model_name / "CHaRS" / f"k{k}_sim{sim}_lambda{lmda}"
            output_layers = output_path / "dirablate_layers"
            output_path.mkdir(parents=True, exist_ok=True)

            with open(output_path / f"nonparam-{data_type}-{language_id}-baseline.json", "w", encoding="utf-8") as f:
                json.dump(baseline_responses, f, indent=4, ensure_ascii=False)

            for ex_pt in hp["extraction_points"]:
                tag = f"r_{ex_pt}" if relocate_mode else str(ex_pt)
                output_layers_ex_pt = output_layers / tag
                logger.info(f"=== Processing extraction point {tag}")

                responses_file = output_layers_ex_pt / f"nonparam_{sim}-{data_type}-{language_id}.json"
                steered_responses = {}
                if responses_file.exists() and not overwrite_mode:
                    with open(responses_file, "r", encoding="utf-8") as f:
                        steered_responses = json.load(f)


                config_files = sorted(
                    output_layers_ex_pt.glob(f"steering_config-{language_id}-*.npy"))

                if not config_files:
                    logger.warning(
                        "No steering config found in %s",
                        output_layers_ex_pt,
                    )
                    continue

                config_file = config_files[0]
                logger.info(f"config file: {config_file}")

                logger.info("Steering Directional Ablation")
                steerer = CHaRSSteerer(model, config_file)
                steered_responses[0] = generate_responses(
                    model, tokenizer, conversations, batch_size, max_new_tokens, steerer=steerer
                )

                with open(responses_file, "w", encoding="utf-8") as f:
                    json.dump(steered_responses, f, indent=4, ensure_ascii=False)
                del steerer

        del model
        torch.cuda.empty_cache()

    print("Total Time Taken:", datetime.datetime.now() - start_time)


if __name__ == "__main__":
    main()
