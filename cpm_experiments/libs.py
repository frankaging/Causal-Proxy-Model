import json
import pickle
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
from models.modelings_gpt2 import *
from models.modelings_lstm import *

from eval_pipeline.models.abstract_model import Model 
from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.utils.data_utils import preprocess_hf_dataset
from eval_pipeline.customized_models.bert import BertForNonlinearSequenceClassification
from eval_pipeline.customized_models.roberta import RobertaForNonlinearSequenceClassification
from eval_pipeline.customized_models.gpt2 import GPT2ForNonlinearSequenceClassification
from eval_pipeline.customized_models.lstm.lstm import LSTMForNonLinearSequenceClassification

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

class CausalExplainer(object):
    def __init__(self):
        pass

def cebab_pipeline(
    model, explainer, 
    train_dataset, dev_dataset, 
    dataset_type='5-way', 
    shorten_model_name=False,
    correction_epsilon=0.001,
):
    # get predictions on train and dev
    train_predictions, _ = model.predict_proba(
        train_dataset
    )
    dev_predictions, dev_report = model.predict_proba(
        dev_dataset
    )

    # append predictions to datasets
    train_dataset['prediction'] = list(train_predictions)
    dev_dataset['prediction'] = list(dev_predictions)

    # fit explainer
    explainer.fit(
        train_dataset, train_predictions, 
        model, dev_dataset
    )

    # get intervention pairs
    
    pairs_dataset = get_intervention_pairs(
        dev_dataset, dataset_type=dataset_type
    )  # TODO why is the index not unique here?
        
    # get explanations
    if isinstance(explainer, CausalExplainer):
        explanations = explainer.estimate_icace(
            pairs_dataset,
            train_dataset # for query data.
        )
    else:
        explanations = explainer.estimate_icace(
            pairs_dataset,
        )
    
    # append explanations to the pairs
    pairs_dataset['EICaCE'] = explanations
    
    # TODO: add cosine
    pairs_dataset = metric_utils._calculate_ite(pairs_dataset)  # effect of crowd-workers on other crowd-workers (no model, no explainer)
    
    def _calculate_icace(pairs):
        """
        This metric measures the effect of a certain concept on the given model.
        It is independent of the explainer.
        """
        pairs['ICaCE'] = (pairs['prediction_counterfactual'] - pairs['prediction_base']).apply(lambda x: np.round(x, decimals=4))

        return pairs
    pairs_dataset = _calculate_icace(pairs_dataset)  # effect of concept on the model (with model, no explainer)
    
    # TOREMOVE: just to try if we ignore all [0,0,...] cases here.
    # zeros_like = tuple([0.0 for i in range(int(dataset_type.split("-")[0]))])
    # pairs_dataset = pairs_dataset[pairs_dataset.ICaCE.map(tuple).isin([zeros_like])==False]
    
    def _cosine_distance(a,b,epsilon):
        if epsilon == None:
            if np.linalg.norm(a, ord=2) == 0 or np.linalg.norm(b, ord=2) == 0:
                return 1
            else:
                return cosine(a,b)
        
        if np.linalg.norm(a, ord=2) == 0 and np.linalg.norm(b, ord=2) == 0:
            """
            We cannot determine whether prediction is corrected or not.
            Thus, we simply return 1.
            """
            return 1
        elif np.linalg.norm(a, ord=2) == 0:
            """
            When true iCACE score is 0, instead of always returning 1, we
            allow some epsilon error by default. If the EiCACE is within a
            range, we return the error as 0.
            """
            if np.max(np.abs(b)) <= epsilon:
                return 0
            else:
                return 1
        elif np.linalg.norm(b, ord=2) == 0:
            """
            This case happens when iCACE is not 0, but EiCACE is 0. This is
            unlikely, but we give score of 1 for this case.
            """
            return 1
        else:
            return cosine(a,b)
    
    def _calculate_estimate_loss(pairs,epsilon):
        """
        Calculate the distance between the ICaCE and EICaCE.
        """

        pairs['ICaCE-L2'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: np.linalg.norm(x[0] - x[1], ord=2), axis=1)
        pairs['ICaCE-cosine'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: _cosine_distance(x[0], x[1], epsilon), axis=1)
        pairs['ICaCE-normdiff'] = pairs[['ICaCE', 'EICaCE']].apply(lambda x: abs(np.linalg.norm(x[0], ord=2) - np.linalg.norm(x[1], ord=2)), axis=1)

        return pairs
    
    pairs_dataset = _calculate_estimate_loss(pairs_dataset,correction_epsilon)  # l2 CEBaB Score (model and explainer)

    # only keep columns relevant for metrics
    CEBaB_metrics_per_pair = pairs_dataset[[
        'intervention_type', 'intervention_aspect_base', 'intervention_aspect_counterfactual', 'ITE', 'ICaCE', 'EICaCE', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']].copy()
    CEBaB_metrics_per_pair['count'] = 1

    # get CEBaB tables
    metrics = ['count', 'ICaCE', 'EICaCE']

    groupby_aspect_direction = ['intervention_type', 'intervention_aspect_base', 'intervention_aspect_counterfactual']

    CaCE_per_aspect_direction = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, metrics)
    CaCE_per_aspect_direction.columns = ['count', 'CaCE', 'ECaCE']
    CaCE_per_aspect_direction = CaCE_per_aspect_direction.set_index(['count'], append=True)
    
    ACaCE_per_aspect = metric_utils._aggregate_metrics(CaCE_per_aspect_direction.abs(), ['intervention_type'], ['CaCE', 'ECaCE'])
    ACaCE_per_aspect.columns = ['ACaCE', 'EACaCE']

    CEBaB_metrics_per_aspect_direction = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])
    CEBaB_metrics_per_aspect_direction.columns = ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']
    CEBaB_metrics_per_aspect_direction = CEBaB_metrics_per_aspect_direction.set_index(['count'], append=True)

    CEBaB_metrics_per_aspect = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, ['intervention_type'], ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])
    CEBaB_metrics_per_aspect.columns = ['count', 'ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff']
    CEBaB_metrics_per_aspect = CEBaB_metrics_per_aspect.set_index(['count'], append=True)

    CEBaB_metrics = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, [], ['ICaCE-L2', 'ICaCE-cosine', 'ICaCE-normdiff'])

    # get ATE table
    ATE = metric_utils._aggregate_metrics(CEBaB_metrics_per_pair, groupby_aspect_direction, ['count', 'ITE'])
    ATE.columns = ['count', 'ATE']

    # add model and explainer information
    if shorten_model_name:
        model_name = str(model).split('.')[0]
    else:
        model_name = str(model)

    CaCE_per_aspect_direction.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) if col != 'CaCE' else (model_name, '', col) for col in CaCE_per_aspect_direction.columns])
    ACaCE_per_aspect.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) if col != 'ACaCE' else (model_name, '', col) for col in ACaCE_per_aspect.columns])
    CEBaB_metrics_per_aspect_direction.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) for col in CEBaB_metrics_per_aspect_direction.columns])
    CEBaB_metrics_per_aspect.columns = pd.MultiIndex.from_tuples(
        [(model_name, str(explainer), col) for col in CEBaB_metrics_per_aspect.columns])
    CEBaB_metrics.index = pd.MultiIndex.from_product([[model_name], [str(explainer)], CEBaB_metrics.index])
    
    # performance report
    performance_report_index = ['macro-f1', 'accuracy']
    performance_report_data = [dev_report['macro avg']['f1-score'], dev_report['accuracy']]
    performance_report_col = [model_name]
    performance_report = pd.DataFrame(data=performance_report_data, index=performance_report_index, columns=performance_report_col)

    return pairs_dataset, ATE, CEBaB_metrics, CEBaB_metrics_per_aspect_direction, CEBaB_metrics_per_aspect, CaCE_per_aspect_direction, ACaCE_per_aspect, performance_report

def intervene_neuron_logits(
    explanator, hidden_reprs, counterfactual_reprs, neuron_id
):
    hidden_reprs[0,neuron_id] = counterfactual_reprs[0,neuron_id]
    intervened_outputs, _, _ = explanator.model.forward_with_cls_hidden_reprs(
        cls_hidden_reprs=hidden_reprs.unsqueeze(dim=1)
    )
    intervened_logits = torch.nn.functional.softmax(
            intervened_outputs.logits[0].cpu(), dim=-1
    ).detach()[0]
    return intervened_logits