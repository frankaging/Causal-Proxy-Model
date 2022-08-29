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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, KLDivLoss, CosineEmbeddingLoss
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import inspect
import datasets
from datasets import Dataset
from sklearn.metrics import classification_report

from models.modelings_abstraction import *
from models.modelings_roberta import *

from utils.optimization import *
from transformers.optimization import AdamW, Adafactor, get_scheduler
from transformers.trainer_pt_utils import (
    get_parameter_names,
)

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, args, wandb_metadata):
        self.args = args
        
        self.last_loss = 0.0
        self.total_loss_epoch = 0.0
        self.n_iter = 0
        self.epoch = 0
        self.n_total_iter = 0
        self.last_log = 0
        self.lr_this_step = 0.0
        self.n_sequences_epoch = 0
        self.n_effective_aspect_sequence_epoch = 0
        self.n_effective_iit_sequence_epoch = 0
        
        # last.
        self.total_loss_epoch = 0
        self.last_loss = 0.0
        self.last_seq_cls_loss = 0.0
        self.last_mul_cls_loss = 0.0
        self.last_iit_cls_loss = 0.0
        
        self.last_seq_cls_acc = 1.0/5
        self.last_mul_cls_acc = 1.0/3
        self.last_iit_cls_acc = 1.0/5
        
        # accumulated.
        self.accumulated_loss = 0.0
        self.accumulated_seq_cls_loss = 0.0
        self.accumulated_mul_cls_loss = 0.0
        self.accumulated_iit_cls_loss = 0.0
        
        self.accumulated_seq_cls_count = 0
        self.accumulated_mul_cls_count = 0
        self.accumulated_iit_cls_count = 0

        # averaged.
        self.averaged_loss = 0.0
        self.averaged_seq_cls_loss = 0.0
        self.averaged_mul_cls_loss = 0.0
        self.averaged_iit_cls_loss = 0.0
        
        self.averaged_seq_cls_acc = 1.0/5
        self.averaged_mul_cls_acc = 1.0/3
        self.averaged_iit_cls_acc = 1.0/5
        
        self.kld_loss_fct = KLDivLoss(reduction="batchmean")
        self.cosine_loss_fct = CosineEmbeddingLoss(reduction="mean")
        self.ce_loss_fct = nn.CrossEntropyLoss()
        
        # evaluation metrics
        self.best_cosine_metric = 99
        self.current_patience = 0
        
        if "wandb" in self.args.report_to:
            import wandb
            run = wandb.init(
                project=wandb_metadata.split(":")[-1], 
                entity=wandb_metadata.split(":")[0],
                name=self.args.run_name,
            )
            wandb.config.update(self.args)
        
        # log to a local file
        log_train = open(os.path.join(self.args.output_dir, 'train_log.txt'), 'w', buffering=1)
        log_eval = open(os.path.join(self.args.output_dir, 'eval_log.txt'), 'w', buffering=1)
        print('epoch,global_steps,step,loss,seq_cls_loss,mul_cls_loss,iit_cls_loss,seq_cls_acc,mul_cls_acc,iit_cls_acc', file=log_train)
        print('epoch,global_steps,cosine_metric', file=log_eval)
        log_train.close()
        log_eval.close()
        
    def _logits_matching_loss(
        self,
        pred_logits,
        actual_logits,
        temperature=2.0,
        loss_mask=None,
    ):
        matching_loss = (
            self.kld_loss_fct(
                nn.functional.log_softmax(pred_logits / temperature, dim=-1),
                nn.functional.softmax(actual_logits / temperature, dim=-1),
            )
            * (temperature) ** 2
        )

        loss_mask = torch.ones(pred_logits.shape[0]).bool().to(self.device) if loss_mask == None else loss_mask
        pred_labels = pred_logits.data.max(1)[1].long()[loss_mask]
        actual_labels = actual_logits.data.max(1)[1].long()[loss_mask]
        correct_count = pred_labels.eq(actual_labels).sum().cpu().item()  
        
        return matching_loss, correct_count
    
    def _abstract_classification_loss(
        self,
        logits, # this may be logits, but lets see...
        labels,
        loss_mask=None,
        labels_as_logits=False,
    ):
        if labels is None:
            return None, 0
        if labels_as_logits:
            labels = labels.data.max(1)[1].long()
        return self._seq_classification_loss(logits, labels, loss_mask)

    def _seq_classification_loss(
        self,
        logits,
        labels,
        loss_mask=None,
    ):
        loss_mask = torch.ones(logits.shape[0]).bool().to(self.device) if loss_mask == None else loss_mask
        loss = self.ce_loss_fct(logits[loss_mask], labels[loss_mask].view(-1))
        
        pred_cls = logits[loss_mask].data.max(1)[1]
        correct_count = pred_cls.eq(labels[loss_mask]).sum().cpu().item()
        
        return loss, correct_count # return the correct count, let the outter loop to determine the rest!
    
    def _mul_classification_loss(
        self,
        mul_logits_0, mul_logits_1, mul_logits_2, mul_logits_3,
        mul_labels,
    ):
        loss_0, count_0 = self._seq_classification_loss(mul_logits_0, mul_labels[:,0], mul_labels[:,0]!=-1)
        loss_1, count_1 = self._seq_classification_loss(mul_logits_1, mul_labels[:,1], mul_labels[:,1]!=-1)
        loss_2, count_2 = self._seq_classification_loss(mul_logits_2, mul_labels[:,2], mul_labels[:,2]!=-1)
        loss_3, count_3 = self._seq_classification_loss(mul_logits_3, mul_labels[:,3], mul_labels[:,3]!=-1)
        mul_count = count_0+count_1+count_2+count_3
        w_0 = count_0*1.0/mul_count if mul_count != 0 else 0.25
        w_1 = count_1*1.0/mul_count if mul_count != 0 else 0.25
        w_2 = count_2*1.0/mul_count if mul_count != 0 else 0.25
        w_3 = count_3*1.0/mul_count if mul_count != 0 else 0.25
        return w_0*loss_0+w_1*loss_1+w_2*loss_2+w_3*loss_3, mul_count
    
    def _record(
        self, n_sample, n_effective_aspect_sequence, n_effective_iit_sequence,
        loss, seq_cls_loss, mul_cls_loss, iit_cls_loss,
        seq_cls_count, mul_cls_count, iit_cls_count,
        # optional
        low_label_shifts_count=0,
        base_source_label_shifts_count=0
    ):
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_seq_cls_loss = seq_cls_loss.mean().item() if self.args.n_gpu > 0 else seq_cls_loss.item()
        self.last_mul_cls_loss = mul_cls_loss.mean().item() if self.args.n_gpu > 0 else mul_cls_loss.item()
        self.last_iit_cls_loss = iit_cls_loss.mean().item() if self.args.n_gpu > 0 else iit_cls_loss.item()
        
        self.last_seq_cls_acc = seq_cls_count*1.0/n_sample
        self.last_mul_cls_acc = mul_cls_count*1.0/n_effective_aspect_sequence
        if n_effective_iit_sequence == 0:
            self.last_iit_cls_acc = 0
        else:
            self.last_iit_cls_acc = iit_cls_count*1.0/n_effective_iit_sequence
        
        # get the accumulated stats for stable perf audits.
        self.accumulated_loss += self.last_loss * n_sample
        
        self.accumulated_seq_cls_loss += self.last_seq_cls_loss * n_sample
        self.accumulated_mul_cls_loss += self.last_mul_cls_loss * n_effective_aspect_sequence
        self.accumulated_iit_cls_loss += self.last_iit_cls_loss * n_effective_iit_sequence
        
        self.accumulated_seq_cls_count += seq_cls_count
        self.accumulated_mul_cls_count += mul_cls_count
        self.accumulated_iit_cls_count += iit_cls_count
     
        
        self.n_sequences_epoch += n_sample
        self.n_effective_aspect_sequence_epoch += n_effective_aspect_sequence
        self.n_effective_iit_sequence_epoch += n_effective_iit_sequence
        
        # get the averaged stats for stable perf audits.
        self.averaged_loss = self.accumulated_loss / self.n_sequences_epoch
        self.averaged_seq_cls_loss = self.accumulated_seq_cls_loss / self.n_sequences_epoch
        self.averaged_mul_cls_loss = self.accumulated_mul_cls_loss / self.n_effective_aspect_sequence_epoch
        if self.n_effective_iit_sequence_epoch == 0:
            self.averaged_iit_cls_loss = 0
        else:
            self.averaged_iit_cls_loss = self.accumulated_iit_cls_loss / self.n_effective_iit_sequence_epoch
        self.averaged_seq_cls_acc = self.accumulated_seq_cls_count / self.n_sequences_epoch
        self.averaged_mul_cls_acc = self.accumulated_mul_cls_count / self.n_effective_aspect_sequence_epoch
        if self.n_effective_iit_sequence_epoch == 0:
            self.averaged_iit_cls_acc = 0
        else:
            self.averaged_iit_cls_acc = self.accumulated_iit_cls_count / self.n_effective_iit_sequence_epoch
        
    def log_tensorboard(self):
        
        log_train = open(os.path.join(self.args.output_dir, 'train_log.txt'), 'a', buffering=1)
        print('{},{},{},{},{},{},{},{},{},{}'.format(
                self.epoch+1, self.n_total_iter, self.n_iter, 
                self.averaged_loss,
                self.averaged_seq_cls_loss, 
                self.averaged_mul_cls_loss,
                self.averaged_iit_cls_loss, 
                self.averaged_seq_cls_acc,
                self.averaged_mul_cls_acc,
                self.averaged_iit_cls_acc
            ),
            file=log_train
        )
        log_train.close()
        
        if "wandb" in self.args.report_to:
            wandb.log(
                {
                    "train/loss": self.averaged_loss, 
                    "train/seq_cls_loss": self.averaged_seq_cls_loss, 
                    "train/mul_cls_loss": self.averaged_mul_cls_loss, 
                    "train/iit_cls_loss": self.averaged_iit_cls_loss, 
                    "train/seq_cls_acc": self.averaged_seq_cls_acc, 
                    "train/mul_cls_acc": self.averaged_mul_cls_acc, 
                    "train/iit_cls_acc": self.averaged_iit_cls_acc, 
                    
                    "train/lr": float(self.lr_this_step),
                    "train/speed": time.time()-self.last_log,
                }, 
                step=self.n_total_iter
            )
        elif "none" in self.args.report_to:
            pass
        
    def _calculate_metrics(
        self,
        actual,
        pred,
    ):
        result = {}
        result_to_print = classification_report(
            actual, pred, digits=5, output_dict=True)
        print(classification_report(actual, pred, digits=5))
        result["accuracy"] = result_to_print["accuracy"]
        result["Macro-F1"] = result_to_print["macro avg"]["f1-score"]
        result["Weighted-Macro-F1"] = result_to_print["weighted avg"]["f1-score"]
        return result
    
    def iter(self):
        self.n_iter += 1
        self.n_total_iter += 1
        
        if self.n_total_iter % self.args.logging_steps == 0:
            self.log_tensorboard()
            self.last_log = time.time()
    
    def optimize(self, loss):

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()   
        self.iter()
        
        if (self.n_iter % self.args.gradient_accumulation_steps == 0) or \
            (self.n_iter*self.train_batch_size >= len(self.train_dataset)):
            self.lr_this_step = self.optimizer.param_groups[0]['lr']
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
        
    def end_epoch(self):
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
        self.n_effective_aspect_sequence_epoch = 0
        self.n_effective_iit_sequence_epoch = 0
        
        self.total_loss_epoch = 0
        self.accumulated_loss = 0.0
        self.accumulated_seq_cls_loss = 0.0
        self.accumulated_mul_cls_loss = 0.0
        self.accumulated_iit_cls_loss = 0.0
        
        self.accumulated_seq_cls_count = 0
        self.accumulated_mul_cls_count = 0
        self.accumulated_iit_cls_count = 0
        self.accumulated_abstract_cls_count = 0
        
    def evaluate(self):
        return False
        # TODO: this function needs to be rebuilt.
        
    def train(self):
        
        self.last_log = time.time()
        is_stopped_early = False
        
        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):
            # prevent end of epoch eval state switch.
            self.low_level_model.model.train()
            train_dataloader = self.get_train_dataloader()
            iter_bar = tqdm(train_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
                # we first evaluate current model, and see if we need early stop.
                if self.n_total_iter != 0 and \
                    self.n_total_iter % self.args.save_steps == 0:
                    if self.evaluate():
                        is_stopped_early = True
                        break
                prepared_batch = self.prepare_batch(batch)
                self._step(*prepared_batch)
                iter_bar.update()
                iter_bar.set_postfix(
                    {
                        "Last_loss": f"{self.last_loss:.2f}", 
                        "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}" if self.n_iter != 0 else 0.0, 
                    }
                )
            if is_stopped_early:
                break
            iter_bar.close()

            logger.info(f"--- Ending epoch {self.epoch}/{self.args.num_train_epochs-1}")
            self.end_epoch()
        
        if is_stopped_early == True:
            logger.info("Training is early stopped as we found the best performing model")
        else:
            logger.info("Training is finished")

    def save_checkpoint(self):
        try:
            self.low_level_model.model.save_pretrained(
                self.args.output_dir,
            )
        except:
            self.low_level_model.model.module.save_pretrained(
                self.args.output_dir,
            )
            
    def _remove_unused_columns(self, dataset, description: Optional[str] = None):
        if not self.args.remove_unused_columns:
            return dataset
        
        # all these columns are related to counterfactuals, interventional data.
        self._signature_columns = [
            "label", "label_ids", "input_ids", "attention_mask", "original_id", "id",
            'ambiance_label', 'food_label', 'noise_label', 'service_label',
        ]
        self._signature_columns += [
            'id_base', 'id_counterfactual', "edit_id_counterfactual", "edit_id_base",
            'intervention_aspect', 'intervention_aspect_label', 
            'input_ids_counterfactual',
            'attention_mask_counterfactual',
            'ambiance_label_base', 'food_label_base', 
            'noise_label_base', 'service_label_base',
            'ambiance_label_counterfactual', 'food_label_counterfactual', 
            'noise_label_counterfactual', 'service_label_counterfactual',
            'logits_base', 'logits_counterfactual', 'logits', 
            'prediction_base', 'icace',
            'input_ids_approximate', 'token_type_ids_approximate', 
            'attention_mask_approximate', 'is_counterfactual_pairs'
        ]

        ignored_columns = list(set(dataset.column_names) - set(self._signature_columns))
    
        if len(ignored_columns) > 0:
            dset_description = "" if description is None else f"in the {description} set "
            logger.info(
                f"The following columns {', '.join(ignored_columns)} in {dset_description} are ignored."
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
            batch_size=self.train_batch_size,
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
            raise ValueError("Trainer: eval requires a eval_dataset.")

        eval_dataset = self.eval_dataset
        eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        
        return DataLoader(
            eval_dataset,
            sampler=SequentialSampler(eval_dataset),
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
        