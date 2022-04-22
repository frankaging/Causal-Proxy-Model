#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
from functools import reduce

import datasets
import numpy as np
from datasets import load_dataset, load_metric, load_from_disk
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

from models.modelings_roberta import *
from IITTrainer import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
task_to_keys = {
    "opentable": ("text", None),
}

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
        default=0.0,
        metadata={
            "help": "Loss coefficient for the multitask objective."}
    )
        
    beta: float = field(
        default=0.0,
        metadata={
            "help": "Loss coefficient for the IIT objective."}
    )
        
    wandb_metadata: str = field(
        default="go:IIT-ABSA",
        metadata={
            "help": "[username]:[project_name]"},
    )
        
    eval_exclude_neutral: bool = field(
        default=False,
        metadata={
            "help": "Whether to exclude neutral class when evaluating."}
    ) 
 
    intervention_h_dim: int = field(
        default=100,
        metadata={
            "help": "Hidden dimension size to interchange per aspect."}
    )
        
    high_level_model_type: str = field(
        default="logistic_regression",
        metadata={
            "help": "How the high level model infer the correct label."}
    )
      
    all_layers: bool = field(
        default=False,
        metadata={
            "help": "Whether to performance interchange intervention at all layers."}
    ) 


# In[ ]:


def main():

    os.environ["TRANSFORMERS_CACHE"] = "../huggingface_cache/"
    os.environ["WANDB_PROJECT"] = "IIT_ABSA"

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
    
    # make the name shorter.
    # overwrite the output dir a little bit.
    data_dir_postfix = data_args.dataset_name.strip("/").split("/")[-1]
    if training_args.do_train:
        sub_output_dir = f"{data_args.task_name}.train.{data_args.train_split_name}"                         f".alpha.{model_args.alpha}.beta.{model_args.beta}"                         f".dim.{model_args.intervention_h_dim}"                         f".hightype.{model_args.high_level_model_type}.{data_dir_postfix}"
    elif training_args.do_eval:
        train_dir = model_args.model_name_or_path.strip("/").split("/")[-1]
        sub_output_dir = f"{train_dir}.eval.{data_args.eval_split_name}.{data_dir_postfix}"
    if training_args.do_train:
        sub_output_dir = f"{sub_output_dir}_seed_{training_args.seed}"
        
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
    if data_args.dataset_name is not None and os.path.isdir(data_args.dataset_name):
        raw_datasets = load_from_disk(
            data_args.dataset_name,
        )
    else:
        raise ValueError(
            "Need a huggingface datasets formatted directory for `dataset_name`.")
        
    label_list = sorted(list(set(raw_datasets["train"]["label"]).union(
        set(raw_datasets["validation"]["label"]))))
        
    num_labels = len(label_list)
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
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
    
    model = IITRobertaForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    
    # Preprocessing the raw_datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

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
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (
                examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding,
                           max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1)
                               for l in examples["label"]]
        return result
        
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    if training_args.do_train:
        train_dataset = raw_datasets[data_args.train_split_name]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(
                range(data_args.max_train_samples))
    
    if training_args.do_eval:
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
    
    low_level_model = InterventionableIITRobertaForSequenceClassification(
        model=model,
        all_layers=model_args.all_layers
    )
    high_level_model = InterventionableAbstractionModelForABSA(
        model=AbstractionModelForABSA(
            model_type=model_args.high_level_model_type,
        ),
    )
    
    # Initialize our Trainer
    trainer = ABSAIITTrainer(
        low_level_model=low_level_model,
        high_level_model=high_level_model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=data_collator,
        device=device,
        alpha=model_args.alpha,
        beta=model_args.beta,
        wandb_metadata=model_args.wandb_metadata,
        eval_exclude_neutral=model_args.eval_exclude_neutral,
        high_level_model_type=model_args.high_level_model_type,
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

