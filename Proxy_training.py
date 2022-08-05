#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from functools import reduce
import copy

import datasets
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk, concatenate_datasets, Dataset, DatasetDict
from sklearn.metrics import classification_report

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from models.modelings_bert import *
from models.modelings_roberta import *
from models.modelings_gpt2 import *
from models.modelings_lstm import *
from ProxyTrainer import *
from eval_pipeline.utils.data_utils import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
task_to_keys = {
    "cebab": ("text", None),
}
label_key = "label"

import logging
logger = logging.getLogger(__name__)


# In[ ]:


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " +
                  ", ".join(task_to_keys.keys())},
    )
    train_split_name: Optional[str] = field(
        default="train",
        metadata={"help": "The name of split this is trained on."},
    )
    eval_split_name: Optional[str] = field(
        default="validation",
        metadata={"help": "The name of split this is evaluated with."},
    )  
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={
                                     "help": "A csv or a json file containing the test data."})
 
    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    high_level_model_name_or_path: str = field(
        default=None, metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
        
    no_gpu: bool = field(
        default=False,
        metadata={
            "help": "Device"}
    ) 
        
    alpha: float = field(
        default=1.0,
        metadata={
            "help": "Loss coefficient for the task objective."}
    )
        
    beta: float = field(
        default=1.0,
        metadata={
            "help": "Loss coefficient for the multitask objective."}
    )
        
    gemma: float = field(
        default=3.0,
        metadata={
            "help": "Loss coefficient for the IIT objective."}
    )

    classifier_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Whether to set dropout on the IIT classifier."}
    ) 
     
    encoder_dropout: float = field(
        default=0.1,
        metadata={
            "help": "Whether to set dropout on the IIT classifier."}
    ) 
       
    wandb_metadata: str = field(
        default="go:IIT-ABSA",
        metadata={
            "help": "[username]:[project_name]"},
    )
    
    k: int = field(
        default=0,
        metadata={
            "help": "In case of training with few-shot of true counterfactuals, "\
                    "we use this field to quantify the number of true "\
                    "counterfactuals we use."}
    ) 
    counterfactual_type: str = field(
        default="true",
        metadata={
            "help": "[true | approximate | mixed]."},
    )
        
    intervention_h_dim: int = field(
        default=100,
        metadata={
            "help": "Hidden dimension size to interchange per aspect."}
    )
    interchange_hidden_layer: int = field(
        default=12,
        metadata={
            "help": "The interchange layer for transformer-based model. Only BERT work now!"}
    )
        
# In[ ]:


def main():

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    os.environ["TRANSFORMERS_CACHE"] = model_args.cache_dir

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # we need to config stuffs based on the model name.
    training_args.save_steps = 999999
    
    # make the name shorter.
    # overwrite the output dir a little bit.
    model_args.high_level_model_type_or_path = model_args.model_name_or_path
    if "roberta-base" in model_args.high_level_model_type_or_path:
        high_type = "roberta-base"
    elif "bert-base-uncased" in model_args.high_level_model_type_or_path:
        high_type = "bert-base-uncased"
    elif "lstm" in model_args.high_level_model_type_or_path:
        high_type = "lstm"
    elif "gpt" in model_args.high_level_model_type_or_path:
        high_type = "gpt2"
    else:
        assert False

    data_dir_postfix = data_args.dataset_name.strip("/").split("/")[-1]
    if training_args.do_train:
        sub_output_dir = f"{data_args.task_name}"\
                         f".alpha.{model_args.alpha}.beta.{model_args.beta}.gemma.{model_args.gemma}"\
                         f".lr.{training_args.learning_rate}"\
                         f".dim.{model_args.intervention_h_dim}"\
                         f".hightype.{high_type}.{data_dir_postfix}"\
                         f".cls.dropout.{model_args.classifier_dropout}"\
                         f".enc.dropout.{model_args.encoder_dropout}.counter.type.{model_args.counterfactual_type}"\
                         f".k.{model_args.k}.int.layer.{model_args.interchange_hidden_layer}"
        
    elif training_args.do_eval:
        train_dir = model_args.model_name_or_path.strip("/").split("/")[-1]
        sub_output_dir = f"{train_dir}.eval.{data_args.eval_split_name}.{data_dir_postfix}"
    if training_args.do_train:
        sub_output_dir = f"{sub_output_dir}.seed_{training_args.seed}"
        
    training_args.output_dir = os.path.join(
        training_args.output_dir, sub_output_dir)
    # let us explicity create the directory.
    is_output_dir_exist = os.path.exists(training_args.output_dir)
    if not is_output_dir_exist:
        # Create a new directory because it does not exist 
        os.makedirs(training_args.output_dir)
        print("The new output directory is created!")
    
    # TODO: add split type for multi/iit?
    training_args.run_name = sub_output_dir
    logger.info(f"WANDB RUN NAME: {training_args.run_name}")
    
    # Log on each process the small summary:
    device = torch.device("cpu") if model_args.no_gpu else torch.device("cuda")
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
            
    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    is_regression = False  # Ours: probably not a regression task?
    if data_args.dataset_name is not None and not os.path.isdir(data_args.dataset_name):
        raw_datasets = load_dataset(
            data_args.dataset_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True, # we may delete this!
        )
    # we should keep using this later, as we want to use the HF dataset!
    elif data_args.dataset_name is not None and os.path.isdir(data_args.dataset_name):
        raw_datasets = load_from_disk(
            data_args.dataset_name,
        )
    else:
        raise ValueError(
            "Need a huggingface datasets formatted directory for `dataset_name`.")
    if "2-class" in model_args.model_name_or_path:
        dataset_type = "2-way"
    elif "3-class" in model_args.model_name_or_path:
        dataset_type = "3-way"
    elif "5-class" in model_args.model_name_or_path:
        dataset_type = "5-way"
    else:
        assert False
    (train_exclusive, train_inclusive), eval_dataset, test_dataset = preprocess_hf_dataset_inclusive(
        raw_datasets, verbose=1, dataset_type=dataset_type
    )
    assert model_args.counterfactual_type in {"true", "approximate"}
    train_dataset, train_pairs_dataset = get_train_singles_and_pairs(
        train_exclusive, train_inclusive, 
        training_args.seed, model_args.k, 
        dataset_type=dataset_type,
        approximate=True if model_args.counterfactual_type == "approximate" else False
    )
    # do some special column filtering and renaming!
    train_columns_to_keep = [
        'original_id', 'edit_id', 'is_original', 
        'description', 'review_majority',
        'food_aspect_majority', 'ambiance_aspect_majority', 
        'service_aspect_majority', 'noise_aspect_majority'
    ]
    aspect_label_encode = {
        "Negative":0,
        "Positive":1,
        "unknown":2,
        "no majority": 2,
    }
    aspect_encode = {
        "ambiance":0,
        "food":1,
        "noise":2,
        "service": 3,
    }
    new_dfs = []
    for _df in [train_dataset, eval_dataset, test_dataset]:
        _df = _df[train_columns_to_keep].rename(
            columns={
                'description': 'text', 
                'review_majority': 'label',
                'food_aspect_majority': 'food_label',
                'ambiance_aspect_majority': 'ambiance_label',
                'service_aspect_majority': 'service_label',
                'noise_aspect_majority': 'noise_label'
            }
        )
        _df = _df.replace("", -1).replace(
            {
                "food_label": aspect_label_encode,
                "ambiance_label": aspect_label_encode,
                "service_label": aspect_label_encode,
                "noise_label": aspect_label_encode
            }
        )
        new_dfs += [_df]
    train_dataset, eval_dataset, test_dataset = new_dfs[0], new_dfs[1], new_dfs[2]
    
    train_pairs_columns_to_keep = [
        "original_id_base", "edit_id_base", "edit_id_counterfactual",
        "description_base", 
        "description_counterfactual", "intervention_type",
        "intervention_aspect_counterfactual",
        'food_aspect_majority_base', 'ambiance_aspect_majority_base', 
        'service_aspect_majority_base', 'noise_aspect_majority_base',
        'food_aspect_majority_counterfactual', 'ambiance_aspect_majority_counterfactual', 
        'service_aspect_majority_counterfactual', 'noise_aspect_majority_counterfactual',
    ]
    train_pairs_dataset = train_pairs_dataset[train_pairs_columns_to_keep].rename(
        columns={
            'description_base': 'text', 
            'review_majority_base': 'label',
            'original_id_base': 'original_id',
            'description_counterfactual': 'text_counterfactual', 
            'intervention_type': 'intervention_aspect',
            'intervention_aspect_counterfactual': 'intervention_aspect_label',
            'food_aspect_majority_base': 'food_label_base',
            'ambiance_aspect_majority_base': 'ambiance_label_base',
            'service_aspect_majority_base': 'service_label_base',
            'noise_aspect_majority_base': 'noise_label_base',
            'food_aspect_majority_counterfactual': 'food_label_counterfactual',
            'ambiance_aspect_majority_counterfactual': 'ambiance_label_counterfactual',
            'service_aspect_majority_counterfactual': 'service_label_counterfactual',
            'noise_aspect_majority_counterfactual': 'noise_label_counterfactual'
        }
    )
    train_pairs_dataset = train_pairs_dataset.replace("", -1).replace(
        {
            "intervention_aspect": aspect_encode,
            "intervention_aspect_label": aspect_label_encode,
            "food_label_base": aspect_label_encode,
            "ambiance_label_base": aspect_label_encode,
            "service_label_base": aspect_label_encode,
            "noise_label_base": aspect_label_encode,
            "food_label_counterfactual": aspect_label_encode,
            "ambiance_label_counterfactual": aspect_label_encode,
            "service_label_counterfactual": aspect_label_encode,
            "noise_label_counterfactual": aspect_label_encode
        }
    )
    train_dataset = Dataset.from_pandas(train_dataset)
    train_pairs_dataset = Dataset.from_pandas(train_pairs_dataset)
    eval_dataset = Dataset.from_pandas(eval_dataset)
    test_dataset = Dataset.from_pandas(test_dataset)
    raw_datasets = DatasetDict()
    raw_datasets['train'] = train_dataset
    raw_datasets['train_pairs'] = train_pairs_dataset
    raw_datasets['validation'] = eval_dataset
    raw_datasets['test'] = test_dataset

    label_list = sorted(list(set(train_dataset[label_key])))
    num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if "lstm" in model_args.model_name_or_path:
        model_args.config_name = "bert-base-uncased"
        model_args.tokenizer_name = "bert-base-uncased"
    elif "bert-base-uncased" in model_args.model_name_or_path:
        model_args.tokenizer_name = "bert-base-uncased"
    elif "roberta-base" in model_args.model_name_or_path:
        model_args.tokenizer_name = "roberta-base"
    elif "gpt2" in model_args.model_name_or_path:
        model_args.tokenizer_name = "gpt2"
        
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.intervention_h_dim = model_args.intervention_h_dim
    assert config.intervention_h_dim*4 <= config.hidden_size
    logger.warning(
        f"Hey, per aspect this is the size you are interchange with: {config.intervention_h_dim}"
    )
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    if "roberta" in model_args.model_name_or_path:
        model_constructor = IITRobertaForSequenceClassification
    elif "bert" in model_args.model_name_or_path:
        model_constructor = IITBERTForSequenceClassification
    elif "gpt" in model_args.model_name_or_path:
        model_constructor = IITGPT2ForSequenceClassification
    elif "lstm" in model_args.moderaw_counterfactual_datasetsl_name_or_path:
        model_constructor = IITLSTMForSequenceClassification
    else:
        raise ValueError(
            "Only support RoBERTa, BERT, GPT2 models.")
    
    low_level_config = copy.deepcopy(config)
    # for the proxy model, we may need to disable
    # the final dropout to maximize the causal abstraction.
    low_level_config.classifier_dropout = model_args.classifier_dropout
    low_level_config.hidden_dropout_prob = model_args.encoder_dropout
    low_level_config.attention_probs_dropout_prob = model_args.encoder_dropout
    # sanity check.
    if "bert-base-uncased" in model_args.high_level_model_type_or_path:
        low_level_config.interchange_hidden_layer = model_args.interchange_hidden_layer
        
    if "lstm" in model_args.model_name_or_path:
        low_level_config.update_embeddings=False
        low_level_config.bidirectional=True
        low_level_config.num_hidden_layers=1
        low_level_config.hidden_size=300
        if os.path.isdir(model_args.model_name_or_path):
            model = model_constructor.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=low_level_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            model = model_constructor(
                config=config,
            )
            # load the preloaded embedding file.
            fasttext_embeddings = torch.load("./models/embeddings.bin")
            model.lstm.embeddings.word_embeddings.weight.data = fasttext_embeddings
    else:
        model = model_constructor.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=low_level_config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
        )
        # some post-editing for customized models.
        if "gpt" in model_args.high_level_model_type_or_path:
            # Define a padding token
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = tokenizer.pad_token_id
            
    low_level_model = InterventionableIITTransformerForSequenceClassification(
        model=model,
    )
    
    if "bert" in model_args.high_level_model_type_or_path or \
        "gpt" in model_args.high_level_model_type_or_path:
        if "roberta" in model_args.high_level_model_type_or_path:
            model_constructor = IITRobertaForSequenceClassification
        elif "bert" in model_args.high_level_model_type_or_path:
            model_constructor = IITBERTForSequenceClassification
        elif "gpt" in model_args.high_level_model_type_or_path:
            model_constructor = IITGPT2ForSequenceClassification
        else:
            raise ValueError(
                "Only support RoBERTa, BERT, GPT2 models.")
        high_level_model = model_constructor.from_pretrained(
                model_args.high_level_model_type_or_path,
                from_tf=bool(".ckpt" in model_args.high_level_model_type_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
        )
        # some post-editing for customized models.
        if "gpt" in model_args.high_level_model_type_or_path:
            # Define a padding token
            high_level_model.config.pad_token_id = tokenizer.pad_token_id
            
        high_level_model = InterventionableIITTransformerForSequenceClassification(
            model=high_level_model,
        )
    elif "lstm" in model_args.model_name_or_path:
        config.update_embeddings=False
        config.bidirectional=True
        config.num_hidden_layers=1
        config.hidden_size=300
        if os.path.isdir(model_args.model_name_or_path):
            high_level_model = IITLSTMForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
            )
        else:
            high_level_model = IITLSTMForSequenceClassification(
                config=config,
            )
            # load the preloaded embedding file.
            fasttext_embeddings = torch.load("./models/embeddings.bin")
            high_level_model.lstm.embeddings.word_embeddings.weight.data = fasttext_embeddings
        high_level_model = InterventionableIITTransformerForSequenceClassification(
            model=high_level_model,
        )
    else:
        assert False

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        # NOTE: priority for saving memory
        padding = False
        
    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(
            num_labels=num_labels).label2id
        and data_args.task_name is not None
        and not is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {
            k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {
                i: int(label_name_to_id[label_list[i]]) for i in range(num_labels)}
        else:
            # TODO: check what should happen here for opentable-multi and -iit
            logger.warning(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None and not is_regression:
        label_to_id = {v: i for i, v in enumerate(label_list)}
        
    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        model.config.label2id = {l: i for i, l in enumerate(label_list)}
        model.config.id2label = {
            id: label for label, id in config.label2id.items()}

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    
    def preprocess_function(examples):
        # Tokenize the texts
        if 'text_counterfactual' in examples:
            result = tokenizer(examples['text'], padding=padding,
                               max_length=max_seq_length, truncation=True)
            result_counterfactual = tokenizer(examples['text_counterfactual'], padding=padding,
                               max_length=max_seq_length, truncation=True)
            result["input_ids_counterfactual"] = result_counterfactual["input_ids"]
            result["token_type_ids_counterfactual"] = result_counterfactual["token_type_ids"]
            result["attention_mask_counterfactual"] = result_counterfactual["attention_mask"]
        else:
            result = tokenizer(examples['text'], padding=padding,
                               max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and label_key in examples:
            result[label_key] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples[label_key]]
        return result
        
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if training_args.do_train:
        train_dataset = raw_datasets["train_pairs"]
        """
        We use the query dataset to pull out approximate counterfactuals,
        or some source examples for doing interchange interventions.
        """
        query_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))
        max_train_samples = len(train_dataset)
        logger.info(
            f"Sample with max_train_samples={max_train_samples}."
            )

    eval_dataset = raw_datasets[data_args.eval_split_name]
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(
            range(data_args.max_eval_samples))
            
    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(
                f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    metric = load_metric("accuracy")
    
    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None
    
    # Initialize our Trainer
    trainer = CausalProxyModelTrainer(
        low_level_model=low_level_model,
        high_level_model=high_level_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        query_dataset=query_dataset,
        data_collator=data_collator,
        device=device,
        alpha=model_args.alpha,
        beta=model_args.beta,
        gemma=model_args.gemma,
        wandb_metadata=model_args.wandb_metadata,
    )
    
    if training_args.do_train:
        logger.info("Hey Zen: Life is sad? Let's go get some drinks.")
        trainer.train()
    
    if training_args.do_eval:
        trainer.evaluate()

# In[ ]:


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()