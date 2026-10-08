#!/usr/bin/env python
# coding: utf-8

# ### Evaluate cls tox



import json

k = 4
rtp_score, count = {}, 0

for model in ['gemma-2-2b', 'Meta-Llama-3-8B', 'Qwen2.5-7B']:
    rtp_score[model] = {}
    for intervention_params in ['mean_ot', 'linear_ot', 'cluster_ot', 'pca_ot', 'identity']:
        rtp_score[model][intervention_params] = []
        for seed in [10, 20, 30]:
            try:
                if intervention_params in ["cluster_ot","pca_ot"]:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_toxicity/rtp_summary.json'
                else:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_toxicity/rtp_summary.json'
                with open(filename, 'r') as file:
                    data = json.load(file)
                    rtp_score[model][intervention_params].append(data['rtp_score'])
            except: 
                print(filename, 'not loaded')
rtp_score




import numpy as np
for model in rtp_score.keys():
    for method in rtp_score[model].keys():
        print(model, method, np.mean(rtp_score[model][method]), np.std(rtp_score[model][method]))


# ### Evaluate 0shot tox



import pandas as pd

k = 4
zeroshot_score, count = {}, 0

for model in ['gemma-2-2b', 'Meta-Llama-3-8B', 'Qwen2.5-7B']:
    zeroshot_score[model] = {}
    for intervention_params in ['mean_ot', 'linear_ot', 'cluster_ot', 'pca_ot', 'identity']:
        zeroshot_score[model][intervention_params] = []
        for seed in [10, 20, 30]:
            try:
                if intervention_params in ["cluster_ot","pca_ot"]:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_0shot/0shot_eval.csv'
                else:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_0shot/0shot_eval.csv'
                data = pd.read_csv(filename)
                zeroshot_score[model][intervention_params].append((data['q0_llm_answer'] == 'Yes').mean())
                # alt: zeroshot_score[model][intervention_params].append((data['cls_pred'] == 'toxic').mean())
            except: 
                print(filename, 'not loaded')
zeroshot_score




import numpy as np
for model in zeroshot_score.keys():
    for method in zeroshot_score[model].keys():
        print(model, method, np.mean(zeroshot_score[model][method]), np.std(zeroshot_score[model][method]))


# ### Evaluate PPL Wiki



import json

k = 4
wiki_score, count = {}, 0

for model in ['gemma-2-2b', 'Meta-Llama-3-8B', 'Qwen2.5-7B']:
    wiki_score[model] = {}
    for intervention_params in ['mean_ot', 'linear_ot', 'cluster_ot', 'pca_ot', 'identity']:
        wiki_score[model][intervention_params] = []
        for seed in [10, 20, 30]:
            try:
                if intervention_params in ["cluster_ot","pca_ot"]:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_toxicity/rtp_summary.json'
                else:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_toxicity/rtp_summary.json'
                with open(filename, 'r') as file:
                    data = json.load(file)
                    wiki_score[model][intervention_params].append(data['perplexity-wikipedia'])
            except: 
                print(filename, 'not loaded')
wiki_score




import numpy as np
for model in wiki_score.keys():
    for method in wiki_score[model].keys():
        print(model, method, np.mean(wiki_score[model][method]), np.std(wiki_score[model][method]))


# ### Evaluate PPL Mistral



import pandas as pd

k = 4
mistral_score, count = {}, 0

for model in ['gemma-2-2b', 'Meta-Llama-3-8B', 'Qwen2.5-7B']:
    mistral_score[model] = {}
    for intervention_params in ['mean_ot', 'linear_ot', 'cluster_ot', 'pca_ot', 'identity']:
        mistral_score[model][intervention_params] = []
        for seed in [10, 20, 30]:
            try:
                if intervention_params in ["cluster_ot","pca_ot"]:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_perplexity/model_perplexity.csv'
                else:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_perplexity/model_perplexity.csv'
                data = pd.read_csv(filename)
                mistral_score[model][intervention_params].append(data['ppl_Mistral-7B-v0.1'].mean())
            except: 
                print(filename, 'not loaded')
mistral_score




import numpy as np
for model in mistral_score.keys():
    for method in mistral_score[model].keys():
        print(model, method, np.mean(mistral_score[model][method]), np.std(mistral_score[model][method]))


# ### Evaluate PPL MMLU



import pickle

k = 4
mmlu_score, count = {}, 0

for model in ['gemma-2-2b', 'Meta-Llama-3-8B', 'Qwen2.5-7B']:
    mmlu_score[model] = {}
    for intervention_params in ['mean_ot', 'linear_ot', 'cluster_ot', 'pca_ot', 'identity']:
        mmlu_score[model][intervention_params] = []
        for seed in [10, 20, 30]:
            try:
                if intervention_params in ["cluster_ot","pca_ot"]:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_eleuther/eleuther.pkl'
                else:
                    filename = f'results/CHaRS/{intervention_params}/seed{seed}/evaluate_eleuther/eleuther.pkl'
                with open(filename, 'rb') as file:
                    data = pickle.load(file)
                    mmlu_score[model][intervention_params].append(data['results']['mmlu']['acc,none'])
            except: 
                print(filename, 'not loaded')
mmlu_score




import numpy as np
for model in mmlu_score.keys():
    for method in mmlu_score[model].keys():
        print(model, method, np.mean(mmlu_score[model][method]), np.std(mmlu_score[model][method]))
