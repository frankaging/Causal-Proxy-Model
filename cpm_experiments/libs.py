import json
import os, sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
import pandas as pd
import datasets
from collections import defaultdict
import numpy as np
import random
from tqdm import tqdm
import copy
from sklearn.manifold import TSNE
from numpy import reshape
import seaborn as sns
from collections import Counter
import itertools

import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer, 
    AutoConfig,
)
from math import ceil

import sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from models.modelings_roberta import *
from models.modelings_bert import *

from eval_pipeline.models.abstract_model import Model 
from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.utils.data_utils import preprocess_hf_dataset
from eval_pipeline.customized_models.bert import BertForNonlinearSequenceClassification
from eval_pipeline.utils import metric_utils, get_intervention_pairs
from eval_pipeline.explainers.random_explainer import RandomExplainer
from eval_pipeline.explainers.conexp import CONEXP
from scipy.spatial.distance import cosine

import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
from scipy.stats import pearsonr
# plt.style.use("ggplot")

plt.rcParams["font.family"] = "DejaVu Serif"
font = {'family' : 'DejaVu Serif',
        'size'   : 20}
plt.rc('font', **font)

def query_with_aspect_label(
    df,
    ambiance="Positive",
    service="Positive",
    noise="Positive",
    food="Positive",
):
    return df[
        (df["ambiance_aspect_majority"]==ambiance)&
        (df["service_aspect_majority"]==service)&
        (df["noise_aspect_majority"]==noise)&
        (df["food_aspect_majority"]==food)
    ]

def flatten_logits(
    df,
    col_names,
):
    flatten_list = []
    for col_name in col_names:
        for row in df[col_name]:
            new_row = [v for v in row]
            new_row.append(col_name)
            flatten_list.append(new_row)
    col_names = [f"feature_{i}" for i in range(len(flatten_list[-1])-1)]
    col_names.append("type")
    df = pd.DataFrame(
        flatten_list, 
        columns=col_names
    )
    return df

def group_logits(
    correlate_results,
    concept_idx,
):
    group_results = {
        "Negative" : [],
        "unknown" : [],
        "Positive" : [],
    }
    for i in range(len(correlate_results)):
        if correlate_results[i][concept_idx] != "":
            logit = correlate_results[i][-1]
            group_results[correlate_results[i][concept_idx]].append(logit)
    return group_results

def subplot_func(ax, results, with_labels=False, with_title=False):
    group_result = group_logits(results, i)
    ax.boxplot(
        np.asarray([
            group_result["Negative"],
            group_result["unknown"],
            group_result["Positive"],
        ], dtype=object), 
        labels=["neg", "unk", "pos"]
    )
    line_x = []
    linx_y = []
    for l in group_result["Negative"]:
        line_x.append(1)
        linx_y.append(l)
    for l in group_result["unknown"]:
        line_x.append(2)
        linx_y.append(l)
    for l in group_result["Positive"]:
        line_x.append(3)
        linx_y.append(l)
    corr, _ = pearsonr(line_x, linx_y)
    corr = round(corr, 2)
    ax.plot(
        np.unique(line_x), np.poly1d(
            np.polyfit(line_x, linx_y, 1)
        )(np.unique(line_x)),
        linestyle='dashed',
        color='red',
        alpha=0.2,
        linewidth=2.5,
        label=f"corr={corr}"
    )
    ax.legend(
        loc="upper right"
    )
    if with_title:
        ax.title.set_text(names[i])
    if not with_labels:
        ax.get_xaxis().set_ticks([])
    
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.xaxis.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)
    ax.yaxis.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)

    ax.set_facecolor("white")

    return ax

def get_iit_examples(df):
    """
    Given a dataframe in the new data scheme, return all intervention pairs.
    """
    # Drop label distribution and worker information.
    columns_to_keep = ['id', 'original_id', 'edit_id', 'is_original', 'edit_goal', 'edit_type', 'description', 'review_majority','food_aspect_majority', 'ambiance_aspect_majority', 'service_aspect_majority', 'noise_aspect_majority']
    columns_to_keep += [col for col in df.columns if 'prediction' in col]
    df = df[columns_to_keep]
    return df

def query_with_aspect_label(
    df,
    ambiance="Positive",
    service="Positive",
    noise="Positive",
    food="Positive",
):
    return df[
        (df["ambiance_aspect_majority"]==ambiance)&
        (df["service_aspect_majority"]==service)&
        (df["noise_aspect_majority"]==noise)&
        (df["food_aspect_majority"]==food)
    ]

def flatten_logits(
    df,
    col_names,
):
    flatten_list = []
    for col_name in col_names:
        for row in df[col_name]:
            new_row = [v for v in row]
            new_row.append(col_name)
            flatten_list.append(new_row)
    col_names = [f"feature_{i}" for i in range(len(flatten_list[-1])-1)]
    col_names.append("type")
    df = pd.DataFrame(
        flatten_list, 
        columns=col_names
    )
    return df

def group_logits(
    correlate_results,
    concept_idx,
):
    group_results = {
        "Negative" : [],
        "unknown" : [],
        "Positive" : [],
    }
    for i in range(len(correlate_results)):
        if correlate_results[i][concept_idx] != "":
            logit = correlate_results[i][-1]
            group_results[correlate_results[i][concept_idx]].append(logit)
    return group_results

def subplot_func(ax, index, results, with_labels=False, with_title=False):
    group_result = group_logits(results, index)
    names = ["ambiance", "food", "noise", "service"]
    ax.boxplot(
        np.asarray([
            group_result["Negative"],
            group_result["unknown"],
            group_result["Positive"],
        ], dtype=object), 
        labels=["neg", "unk", "pos"]
    )
    line_x = []
    linx_y = []
    for l in group_result["Negative"]:
        line_x.append(1)
        linx_y.append(l)
    for l in group_result["unknown"]:
        line_x.append(2)
        linx_y.append(l)
    for l in group_result["Positive"]:
        line_x.append(3)
        linx_y.append(l)
    corr, _ = pearsonr(line_x, linx_y)
    corr = round(corr, 2)
    ax.plot(
        np.unique(line_x), np.poly1d(
            np.polyfit(line_x, linx_y, 1)
        )(np.unique(line_x)),
        linestyle='dashed',
        color='red',
        alpha=0.2,
        linewidth=2.5,
        label=f"corr={corr}"
    )
    ax.legend(
        loc="upper right"
    )
    if with_title:
        ax.title.set_text(names[index])
    if not with_labels:
        ax.get_xaxis().set_ticks([])
    
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["top"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.xaxis.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)
    ax.yaxis.grid(color='grey', linestyle='-.', linewidth=1, alpha=0.5)

    ax.set_facecolor("white")

    return ax