TASKS=(
    # tinyBenchmarks
    tinyHellaswag
    tinyArc
    tinyMMLU
    tinyTruthfulQA
    tinyWinogrande
    )

# Define models with their corresponding ports
declare -A MODEL_PORTS=(
    ["Qwen/Qwen2.5-3B-Instruct"]=9901
    ["Qwen/Qwen2.5-7B-Instruct"]=9902
    ["Qwen/Qwen2.5-14B-Instruct"]=9903
    ["meta-llama/Llama-3.2-3B-Instruct"]=9904
    ["meta-llama/Llama-3.1-8B-Instruct"]=9905
    ["google/gemma-2-9b-it"]=9906
    ["Qwen/Qwen2.5-32B-Instruct"]=9907
    ["google/gemma-2-27b-it"]=9908
    ["Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment"]=9909

)

MODELS=(
    Qwen/Qwen2.5-3B-Instruct
    # Qwen/Qwen2.5-7B-Instruct
    # Qwen/Qwen2.5-14B-Instruct
    # Qwen/Qwen2.5-32B-Instruct
    # meta-llama/Llama-3.2-3B-Instruct
    # meta-llama/Llama-3.1-8B-Instruct
    # google/gemma-2-9b-it
    # google/gemma-2-27b-it
    # Unispac/Gemma-2-9B-IT-With-Deeper-Safety-Alignment
)



# ActAdd
declare -A actadd_optimal_layers
actadd_optimal_layers["Qwen/Qwen2.5-3B-Instruct"]="47"
actadd_optimal_layers["Qwen/Qwen2.5-7B-Instruct"]="35"
actadd_optimal_layers["Qwen/Qwen2.5-14B-Instruct"]="47"
actadd_optimal_layers["Qwen/Qwen2.5-32B-Instruct"]="91"
actadd_optimal_layers["meta-llama/Llama-3.1-8B-Instruct"]="25"
actadd_optimal_layers["meta-llama/Llama-3.2-3B-Instruct"]="28"
actadd_optimal_layers["google/gemma-2-9b-it"]="44"

declare -A actadd_optimal_alpha
actadd_optimal_alpha["Qwen/Qwen2.5-3B-Instruct"]="2.3"
actadd_optimal_alpha["Qwen/Qwen2.5-7B-Instruct"]="2.9"
actadd_optimal_alpha["Qwen/Qwen2.5-14B-Instruct"]="5.4"
actadd_optimal_alpha["Qwen/Qwen2.5-32B-Instruct"]="3.4"
actadd_optimal_alpha["meta-llama/Llama-3.1-8B-Instruct"]="1.6"
actadd_optimal_alpha["meta-llama/Llama-3.2-3B-Instruct"]="1.8"
actadd_optimal_alpha["google/gemma-2-9b-it"]="5.1"

# Directional Ablation
declare -A dirablate_optimal_layers
dirablate_optimal_layers["Qwen/Qwen2.5-3B-Instruct"]="46"
dirablate_optimal_layers["Qwen/Qwen2.5-7B-Instruct"]="36"
dirablate_optimal_layers["Qwen/Qwen2.5-14B-Instruct"]="50"
dirablate_optimal_layers["Qwen/Qwen2.5-32B-Instruct"]="90"
dirablate_optimal_layers["meta-llama/Llama-3.1-8B-Instruct"]="23"
dirablate_optimal_layers["meta-llama/Llama-3.2-3B-Instruct"]="24"
dirablate_optimal_layers["google/gemma-2-9b-it"]="48"

# Choice of Cluster (ActAdd)
declare -A actadd_cluster_choice
actadd_cluster_choice["Qwen/Qwen2.5-3B-Instruct"]="15"
actadd_cluster_choice["Qwen/Qwen2.5-7B-Instruct"]="5"
actadd_cluster_choice["Qwen/Qwen2.5-14B-Instruct"]="5"
actadd_cluster_choice["Qwen/Qwen2.5-32B-Instruct"]="15"
actadd_cluster_choice["meta-llama/Llama-3.1-8B-Instruct"]="9"
actadd_cluster_choice["meta-llama/Llama-3.2-3B-Instruct"]="4"
actadd_cluster_choice["google/gemma-2-9b-it"]="10"

# Choice of Cluster (DirAblate)
declare -A dirablate_cluster_choice
dirablate_cluster_choice["Qwen/Qwen2.5-3B-Instruct"]="11"
dirablate_cluster_choice["Qwen/Qwen2.5-7B-Instruct"]="9"
dirablate_cluster_choice["Qwen/Qwen2.5-14B-Instruct"]="6"
dirablate_cluster_choice["Qwen/Qwen2.5-32B-Instruct"]="9"
dirablate_cluster_choice["meta-llama/Llama-3.1-8B-Instruct"]="5"
dirablate_cluster_choice["meta-llama/Llama-3.2-3B-Instruct"]="8"
dirablate_cluster_choice["google/gemma-2-9b-it"]="9"

# No of PC (ActAdd)
declare -A actadd_pc_choice
actadd_pc_choice["Qwen/Qwen2.5-3B-Instruct"]="12"
actadd_pc_choice["Qwen/Qwen2.5-7B-Instruct"]="1"
actadd_pc_choice["Qwen/Qwen2.5-14B-Instruct"]="1"
actadd_pc_choice["Qwen/Qwen2.5-32B-Instruct"]="9"
actadd_pc_choice["meta-llama/Llama-3.1-8B-Instruct"]="1"
actadd_pc_choice["meta-llama/Llama-3.2-3B-Instruct"]="3"
actadd_pc_choice["google/gemma-2-9b-it"]="12"

# Choice of Cluster (DirAblate)
declare -A dirablate_pc_choice
dirablate_pc_choice["Qwen/Qwen2.5-3B-Instruct"]="3"
dirablate_pc_choice["Qwen/Qwen2.5-7B-Instruct"]="6"
dirablate_pc_choice["Qwen/Qwen2.5-14B-Instruct"]="1"
dirablate_pc_choice["Qwen/Qwen2.5-32B-Instruct"]="3"
dirablate_pc_choice["meta-llama/Llama-3.1-8B-Instruct"]="3"
dirablate_pc_choice["meta-llama/Llama-3.2-3B-Instruct"]="12"
dirablate_pc_choice["google/gemma-2-9b-it"]="9"


# PC Steer Relocate Mode
new_adaptive_mode=0
actadd_relocate_mode=1
dirablate_relocate_mode=0

sim=adaptive_gaussian
lmda=adaptive
max_pc=15
inf_sim=adaptive_gaussian

# dir_id=max_sim
LM_EVAL_BIN="${LM_EVAL_BIN:-lm_eval}"
BASE_URL_HOST="${BASE_URL_HOST:-0.0.0.0}"


for model_id in "${MODELS[@]}"; do
    port=${MODEL_PORTS[$model_id]}
    for task in "${TASKS[@]}"; do
        # baseline
        model_name=$(echo "$model_id" | cut -d'/' -f2)
        # echo "(Baseline) Evaluating model: $model_id with angle none on port $port"
        # lm_eval \
        #     --model local-completions \
        #     --tasks ${task} \
        #     --batch_size 1 \
        #     --model_args model=${model_id},base_url=http://0.0.0.0:${port}/angular_steering/none/${ang_str_relocate_mode}/${ang_str_adaptive_mode},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
        #     --output_path ./benchmarks/trad_baseline/${dir_id}/${task}/${model_name}/new_adaptive_${ang_str_adaptive_mode}/adaptive_none \
        #     --log_samples

        optimal_layer_actadd="${actadd_optimal_layers[$model_id]}"
        optimal_alpha_actadd="${actadd_optimal_alpha[$model_id]}"
        optimal_layer_dirablate="${dirablate_optimal_layers[$model_id]}"
        optimal_cc_actadd="${actadd_cluster_choice[$model_id]}"
        optimal_cc_dirablate="${dirablate_cluster_choice[$model_id]}"
        optimal_pc_actadd="${actadd_pc_choice[$model_id]}"
        optimal_pc_dirablate="${dirablate_pc_choice[$model_id]}"

        # Base ActAdd
        echo "(ActAdd) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/cluststeer/${optimal_layer_actadd}/${optimal_alpha_actadd}/${actadd_relocate_mode}/${new_adaptive_mode}/1/${sim}/${lmda}/${inf_sim},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/cluststeer_final/k1_sim${sim}_lambda${lmda}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_actadd}/actadd \
            --log_samples

        # Base Directional Ablation
        echo "(Directional Ablation) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/cluststeer_dirablate/${optimal_layer_dirablate}/${dirablate_relocate_mode}/${new_adaptive_mode}/1/${sim}/${lmda}/${inf_sim},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/cluststeer_final/k1_sim${sim}_lambda${lmda}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_dirablate}/dirablate \
            --log_samples

        # Cluster Steering ActAdd
        echo "(ClustSteer ActAdd) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/cluststeer/${optimal_layer_actadd}/${optimal_alpha_actadd}/${actadd_relocate_mode}/${new_adaptive_mode}/${optimal_cc_actadd}/${sim}/${lmda}/${inf_sim},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/cluststeer_final/k${optimal_cc_actadd}_sim${sim}_lambda${lmda}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_actadd}/actadd \
            --log_samples

        # Cluster Steering DirAblate
        echo "(ClustSteer Directional Ablation) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/cluststeer_dirablate/${optimal_layer_dirablate}/${dirablate_relocate_mode}/${new_adaptive_mode}/${optimal_cc_dirablate}/${sim}/${lmda}/${inf_sim},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/cluststeer_final/k${optimal_cc_dirablate}_sim${sim}_lambda${lmda}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_dirablate}/dirablate \
            --log_samples

        # PC Steering ActAdd
        echo "(PCSteer ActAdd) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/pcsteer/${optimal_layer_actadd}/${optimal_alpha_actadd}/${actadd_relocate_mode}/${new_adaptive_mode}/${optimal_cc_actadd}/${sim}/${lmda}/${max_pc}/${inf_sim}/${optimal_pc_actadd},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/pcsteer_final/k${optimal_cc_actadd}_sim${sim}_lambda${lmda}_maxpc${max_pc}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_actadd}/actadd/${optimal_pc_actadd} \
            --log_samples

        # PC Steering DirAblate
        echo "(PCSteer Directional Ablation) Evaluating model: $model_id"
        "$LM_EVAL_BIN" \
            --model local-completions \
            --tasks ${task} \
            --batch_size 1 \
            --model_args model=${model_id},base_url=http://${BASE_URL_HOST}:${port}/pcsteer_dirablate/${optimal_layer_dirablate}/${dirablate_relocate_mode}/${new_adaptive_mode}/${optimal_cc_dirablate}/${sim}/${lmda}/${max_pc}/${inf_sim}/${optimal_pc_dirablate},num_concurrent=1,max_retries=3,tokenized_requests=False,max_gen_toks=4096 \
            --output_path ./benchmarks/pcsteer_final/k${optimal_cc_dirablate}_sim${sim}_lambda${lmda}_maxpc${max_pc}/${task}/${model_name}/${inf_sim}/new_adaptive_${new_adaptive_mode}/${optimal_layer_dirablate}/dirablate/${optimal_pc_dirablate} \
            --log_samples

    done
done
