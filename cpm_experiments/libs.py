import json
import pickle
import os, sys
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), './modelings/'))
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

from cpm import *

from eval_pipeline.models.abstract_model import Model 
from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.utils.data_utils import *
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

def cebab_pipeline(
    model, explainer, 
    train_dataset, dev_dataset, 
    seed, k, dataset_type='5-way', 
    shorten_model_name=False, 
    train_setting='exclusive', approximate=False
):
    # TODO: add inclusive
    ## k training pairs (sample or get them from a pre-loaded sampled file?)
    ### How? k pairs == 2*k samples, with a maximum of k u's
    ## n-k training singles

    if train_setting in ['inclusive', 'approximate']:
        # NOTE: this can be moved to an outer loop for speed optimization.
        # NOTE: this should be done before the runs and just saved in some files
        # TODO: approx true
        train_dataset, train_pairs_dataset = get_train_singles_and_pairs(
            train_dataset[0], train_dataset[1], 
            seed, k, dataset_type=dataset_type, 
            approximate=approximate
        )
    elif train_setting == 'exclusive':
        pass
 
    # NOTE: we will only work with models that are fitted
    # fit model
    model.fit(train_dataset)
    
    # get predictions on train and dev
    train_predictions, _ = model.predict_proba(train_dataset)
    dev_predictions, dev_report = model.predict_proba(dev_dataset)

    if train_setting in ['inclusive', 'approximate']:
        # TODO: add predictions to dataset
        # TODO: get the model predictions in a pair format for inclusive explainers
        predictions = pd.DataFrame(data=zip(train_dataset['id'].to_numpy(), train_predictions), columns=['id', 'prediction'])

        train_dataset = train_dataset.merge(predictions, on='id')

        predictions_base = predictions.rename(lambda x: x+'_base', axis=1) 
        predictions_counterfactual = predictions.rename(lambda x: x+'_counterfactual', axis=1) 

        train_pairs_dataset = train_pairs_dataset.merge(predictions_base, on='id_base')
        train_pairs_dataset = train_pairs_dataset.merge(predictions_counterfactual, on='id_counterfactual')

    # append predictions to datasets
    # train_dataset['prediction'] = list(train_predictions)
    dev_dataset['prediction'] = list(dev_predictions)

    # get intervention pairs
    # TODO: approx false
    pairs_dataset = get_intervention_pairs(
        dev_dataset, dataset_type=dataset_type
    )  # TODO why is the index not unique here?

    # fit explainer
    # TODO: add inclusive
    if train_setting in ['inclusive', 'approximate']:
        explainer.fit(train_pairs_dataset, train_dataset, model, dev_dataset=pairs_dataset)
    elif train_setting == 'exclusive':
        explainer.fit(train_dataset, train_predictions, model, dev_dataset=dev_dataset)

    # mitigate possible data leakage
    allowed_columns = [
        'description_base',
        'review_majority_base',
        'food_aspect_majority_base',
        'service_aspect_majority_base',
        'noise_aspect_majority_base',
        'ambiance_aspect_majority_base',
        'intervention_type',
        'intervention_aspect_base',
        'intervention_aspect_counterfactual',
        'opentable_metadata_base',
        'prediction_base'
    ]

    pairs_dataset_no_leakage = pairs_dataset.copy()[allowed_columns]

    # get explanations
    if isinstance(explainer, CausalExplainer):
        explanations = explainer.estimate_icace(
            pairs_dataset,
            train_dataset # for query data.
        )
        # we also overwrite the dev_report to use cpm model.
        _, dev_report = explainer.predict_proba(
            dev_dataset
        )
    else:
        explanations = explainer.estimate_icace(
            pairs_dataset,
        )
    
    # append explanations to the pairs
    pairs_dataset['EICaCE'] = explanations

    pairs_dataset = metric_utils._calculate_ite(pairs_dataset)  # effect of crowd-workers on other crowd-workers (no model, no explainer)
    pairs_dataset = metric_utils._calculate_icace(pairs_dataset)  # effect of concept on the model (with model, no explainer)
    pairs_dataset = metric_utils._calculate_estimate_loss(pairs_dataset)  # l2 CEBaB Score (model and explainer)

    # only keep columns relevant for metrics
    CEBaB_metrics_per_pair = pairs_dataset[[
        'intervention_type', 'intervention_aspect_base', 
        'intervention_aspect_counterfactual', 'ITE', 
        'ICaCE', 'EICaCE', 'ICaCE-L2', 'ICaCE-cosine', 
        'ICaCE-normdiff'
    ]].copy()
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

    return pairs_dataset, ATE, CEBaB_metrics, CEBaB_metrics_per_aspect_direction, \
        CEBaB_metrics_per_aspect, CaCE_per_aspect_direction, \
        ACaCE_per_aspect, performance_report
    
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