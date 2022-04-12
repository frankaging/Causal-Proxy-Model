import os
import random
import pickle
import time
import psutil
import wandb

import numpy as np
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from tqdm import tqdm, trange
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import inspect
import datasets

from models.modelings_abstraction import *
from models.modelings_roberta import *

from utils import logging
logger = logging.get_logger(__name__)

class ABSAIITTrainer:
    def __init__(
        self, 
        low_level_model, 
        high_level_model,
        args, 
        train_dataset, 
        eval_dataset, 
        data_collator,

    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.low_level_model = low_level_model
        self.high_level_model = high_level_model
        self.data_collator = data_collator
        
        self._signature_columns = None
        
        self.last_loss = 0.0
        self.total_loss_epoch = 0.0
        self.n_iter = 0
        self.epoch = 0
        self.n_total_iter = 0
        self.last_log = 0
        
    def prepare_batch(
        self,
        input_ids, attention_mask, labels, aspect_labels,
    ):
        # TODO: explore other pairing strategies?
        source_input_ids = input_ids.clone()
        source_attention_mask = attention_mask.clone()
        source_labels = labels.clone()
        source_aspect_labels = aspect_labels.clone()
        
        _sort_index = [i for i in range(source_input_ids.shape[0])]
        random.shuffle(_sort_index)

        source_input_ids = source_input_ids[_sort_index]
        source_attention_mask = source_attention_mask[_sort_index]
        source_labels = source_labels[_sort_index]
        source_aspect_labels = source_aspect_labels[_sort_index]
        
        return input_ids, attention_mask, labels, aspect_labels, \
            source_input_ids, source_attention_mask, source_labels, source_aspect_labels
    
    def train(self):
        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
    
        self.high_level_model.model.eval()
    
        self.last_log = time.time()
        
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            # prevent end of epoch eval state switch.
            self.low_level_model.model.train()
            train_dataloader = self.get_train_dataloader()
            iter_bar = tqdm(train_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                # send to device
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
                aspect_labels = torch.stack(
                    [
                        batch["ambiance_label"], batch["food_label"], 
                        batch["noise_label"], batch["service_label"]
                    ], 
                    dim=-1
                )
                prepared_batch = self.prepare_batch(
                    *(input_ids, attention_mask, labels, aspect_labels)
                )
                self.step(
                    *prepared_batch
                )
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}", 
                    }
                )
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.args.num_train_epochs-1}")
            self.end_epoch()

        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Training is finished")
    
    def step(
        self,
        base_input_ids, base_attention_mask, 
        base_labels, base_aspect_labels,
        source_input_ids, source_attention_mask, 
        source_labels, source_aspect_labels,
        skip_update_iter=False
    ):
        loss = 0.0
        
        self.optimize(loss, skip_update_iter=skip_update_iter)
    
    def optimize(self, loss, skip_update_iter=False):
        
        if not skip_update_iter:
            self.iter()
    
    def iter(self):
        self.n_iter += 1
        self.n_total_iter += 1
        if self.n_total_iter % self.args.save_steps == 0:
            pass
            # you can uncomment this line, if you really have checkpoints.
            # self.save_checkpoint()
        
        """
        Logging is not affected by the flag skip_update_iter.
        We want to log crossway effects, and losses should be
        in the same magnitude.
        """
        if self.n_total_iter % self.args.logging_steps == 0:
            self.log_tensorboard()
            self.last_log = time.time()
    
    def log_tensorboard(self):
        pass
    
    def end_epoch(self):
        pass
    
    def save_checkpoint(self, checkpoint_name=None):
        pass
    
    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.low_level_model.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # Labels may be named label or label_ids, the default data collator handles that.
            self._signature_columns += [
                "label", "label_ids",
                'ambiance_label', 'food_label', 'noise_label', 'service_label'
            ]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
    
        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {dset_description} don't have a corresponding argument in "
                f"`{self.low_level_model.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
                f" If {', '.join(ignored_columns)} are not expected by `{self.low_level_model.model.__class__.__name__}.forward`, "
                f" you can safely ignore this message."
            )

        columns = [k for k in self._signature_columns if k in dataset.column_names]
    
        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"], columns=columns, format_kwargs=dataset.format["format_kwargs"]
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)
    
    def get_train_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        train_dataset = self._remove_unused_columns(train_dataset, description="training")

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=RandomSampler(train_dataset),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_eval_dataloader(self):
        """
        Returns the training [`~torch.utils.data.DataLoader`].
        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.eval_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        eval_dataset = self.eval_dataset
        eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")

        return DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=self.args.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    