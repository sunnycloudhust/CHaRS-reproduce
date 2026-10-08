#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import json

import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import StrMethodFormatter




style = "sketch"
model = "FLUX.1-dev"




clip_data_clust = pd.read_csv(f't2i_results/CHaRS/cluster_ot_FLUX.1-dev_{style}/seed1/calculate_clip_score/clip_score.csv', index_col=0)
print(len(clip_data_clust))
clip_data_clust.head()




clip_data_linear = pd.read_csv(f't2i_results/CHaRS/linear_ot_FLUX.1-dev_{style}/seed1/calculate_clip_score/clip_score.csv', index_col=0)
print(len(clip_data_linear))
clip_data_linear.head()




# fig = go.Figure()
cut_off_bound = 0.49
lbs = clip_data_linear['strength'].unique()

zeroshot_mean_ot_cluster = [clip_data_clust[
    (clip_data_clust['strength'] == lb) & (clip_data_clust['conditional_zero_shot_score'] >= cut_off_bound)
].shape[0] / clip_data_clust[clip_data_clust['strength'] == 1].shape[0] for lb in lbs]

clipscore_mean_ot_cluster = [clip_data_clust['conditional_similarity'][
    (clip_data_clust['strength'] == lb) & (clip_data_clust['unconditional_similarity'] >= 0.0)
].sum() / clip_data_clust[clip_data_clust['strength'] == 1].shape[0] for lb in lbs]

df_cluster = pd.DataFrame({
    "strength": lbs,
    "0shot_cluster": zeroshot_mean_ot_cluster,
    "clipscore_cluster": clipscore_mean_ot_cluster,
})
fig, (ax1_left, ax2_left) = plt.subplots(1, 2, figsize=(12.8, 5.6))
sns.lineplot(
    data = df_cluster,
    x = "strength",
    y = "0shot_cluster",
    ax=ax1_left,
    marker="o",
    linewidth=2,
    label = "0-Shot Classification Score",
    color = "#2CA02C"
)

ax1_right = ax1_left.twinx() # Creates a second axes that shares the same x-axis
sns.lineplot(
    data = df_cluster,
    x = "strength",
    y = "clipscore_cluster",
    ax=ax1_right,
    marker="o",
    linewidth=2,
    label = "CLIPScore",
    color = "#1F77B4"
)

zeroshot_mean_ot_linear = [clip_data_linear[
    (clip_data_linear['strength'] == lb) & (clip_data_linear['conditional_zero_shot_score'] >= cut_off_bound)
].shape[0] / clip_data_linear[clip_data_linear['strength'] == 1].shape[0] for lb in lbs]

clipscore_mean_ot_linear = [clip_data_linear['conditional_similarity'][
    (clip_data_linear['strength'] == lb) & (clip_data_linear['unconditional_similarity'] >= 0.0)
].sum() / clip_data_linear[clip_data_linear['strength'] == 1].shape[0] for lb in lbs]

df_linear = pd.DataFrame({
    "strength": lbs,
    "0shot_cluster": zeroshot_mean_ot_linear,
    "clipscore_cluster": clipscore_mean_ot_linear,
})
sns.lineplot(
    data = df_linear,
    x = "strength",
    y = "0shot_cluster",
    ax=ax2_left,
    marker="o",
    linewidth=2,
    label = "0-Shot Classification Score",
    color = "#2CA02C"
)

ax2_right = ax2_left.twinx() # Creates a second axes that shares the same x-axis
sns.lineplot(
    data = df_linear,
    x = "strength",
    y = "clipscore_cluster",
    ax=ax2_right,
    marker="o",
    linewidth=2,
    label = "CLIPScore",
    color = "#1F77B4"
)

# Set labels, label sizes
ax1_left.set_xlabel(r"Strength $\lambda$", fontsize = 22)
ax1_left.set_ylabel("0-Shot Classification Score", fontsize = 22)
ax1_right.set_ylabel("CLIPScore", fontsize = 22)
ax2_left.set_xlabel(r"Strength $\lambda$", fontsize = 22)
ax2_left.set_ylabel("0-Shot Classification Score", fontsize = 22)
ax2_right.set_ylabel("CLIPScore", fontsize = 22)

# Set y axis limits
ax1_left.set_ylim(0.5, 1.05)
ax1_right.set_ylim(0.1, 0.3)
ax2_left.set_ylim(0.5, 1.05)
ax2_right.set_ylim(0.1, 0.3)

# Set ticks, tick sizes
plt.xticks(fontsize=18)
ax1_left.tick_params(labelsize=18)
ax1_right.tick_params(labelsize=18)
ax2_left.tick_params(labelsize=18)
ax2_right.tick_params(labelsize=18)

# Set log scale
ax1_left.set_xscale('log')
ax2_left.set_xscale('log')
ax1_left.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
ax2_left.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))

# Set combined legend for both y axes
lines1, labels1 = ax1_left.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
lines3, labels3 = ax2_left.get_legend_handles_labels()
lines4, labels4 = ax2_right.get_legend_handles_labels()
fig.legend(lines1+lines2, labels1+labels2, loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=20, ncol=2)
for a in [ax1_left, ax1_right, ax2_left, ax2_right]:
    if a.get_legend(): a.get_legend().remove()

# Other grid parameters
ax1_left.grid(True, which="major", linewidth=0.6, alpha=0.7)
ax2_left.grid(True, which="major", linewidth=0.6, alpha=0.7)
plt.tight_layout()
plt.savefig("clipscore_plots/both_stop10_logscale.pdf", bbox_inches='tight')

plt.show()
