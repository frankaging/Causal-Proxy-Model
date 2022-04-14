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
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import inspect
import datasets
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

class ABSAIITTrainer:
    def __init__(
        self, 
        low_level_model, 
        high_level_model,
        args, 
        train_dataset, 
        eval_dataset, 
        data_collator,
        device,
        alpha, beta,
        wandb_metadata,
        eval_exclude_neutral=False,
    ):
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.low_level_model = low_level_model
        self.high_level_model = high_level_model
        self.data_collator = data_collator
        
        self.alpha = alpha
        self.beta = beta
        self.eval_exclude_neutral = eval_exclude_neutral
        
        # device
        self.device = device
        self.low_level_model.model.to(self.device)
        if self.args.n_gpu > 1:
            self.low_level_model.model = torch.nn.DataParallel(self.low_level_model.model)
        
        self._signature_columns = None
        
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
        
        if "wandb" in self.args.report_to:
            import wandb
            run = wandb.init(
                project=wandb_metadata.split(":")[-1], 
                entity=wandb_metadata.split(":")[0],
                name=self.args.run_name,
            )
            wandb.config.update(self.args)
        
        if self.args.n_gpu > 1:
            self.train_batch_size = args.per_device_train_batch_size * self.args.n_gpu
            self.eval_batch_size = args.per_device_eval_batch_size * self.args.n_gpu
        else:
            self.train_batch_size = args.per_device_train_batch_size
            self.eval_batch_size = args.per_device_eval_batch_size
        
        if self.args.do_train:
            num_train_optimization_steps = math.ceil(len(train_dataset) / self.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", len(train_dataset))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            self.num_train_optimization_steps = num_train_optimization_steps
        
        if self.args.do_eval:
            # Run prediction for full data
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", len(eval_dataset))
            logger.info("  Batch size = %d", self.eval_batch_size)
        
        # getting params to optimize early
        decay_parameters = get_parameter_names(self.low_level_model.model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.low_level_model.model.named_parameters() if n in decay_parameters],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.low_level_model.model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer_kwargs = {"lr": self.args.learning_rate}
        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        optimizer_kwargs.update(adam_kwargs)
        self.optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
        self.lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.args.get_warmup_steps(num_train_optimization_steps),
            num_training_steps=num_train_optimization_steps,
        )
        self.lr_this_step = self.optimizer.param_groups[0]['lr']

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
        
        self.accumulated_seq_cls_acc = 1.0/5
        self.accumulated_mul_cls_acc = 1.0/3
        self.accumulated_iit_cls_acc = 1.0/5
        
        # averaged.
        self.averaged_loss = 0.0
        self.averaged_seq_cls_loss = 0.0
        self.averaged_mul_cls_loss = 0.0
        self.averaged_iit_cls_loss = 0.0
        
        self.averaged_seq_cls_acc = 1.0/5
        self.averaged_mul_cls_acc = 1.0/3
        self.averaged_iit_cls_acc = 1.0/5
        
        # log to a local file
        log_train = open(os.path.join(self.args.output_dir, 'train_log.txt'), 'w', buffering=1)
        log_eval = open(os.path.join(self.args.output_dir, 'eval_log.txt'), 'w', buffering=1)
        print('epoch,global_steps,step,loss,seq_cls_loss,mul_cls_loss,iit_cls_loss,seq_cls_acc,mul_cls_acc,iit_cls_acc', file=log_train)
        print('epoch,loss,seq_cls_loss,mul_cls_loss,iit_cls_loss,seq_cls_acc,mul_cls_acc,iit_cls_acc', file=log_eval)
        log_train.close()
        log_eval.close()
        
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
    
    def _calculate_metrics(
        self,
        actual,
        pred,
    ):
        result = {}
        if self.eval_exclude_neutral:
            result_to_print = classification_report(
                actual, pred, digits=5, output_dict=True, labels=[0,1,3,4])
            print(classification_report(actual, pred, digits=5, labels=[0,1,3,4]))
        else:
            result_to_print = classification_report(
                actual, pred, digits=5, output_dict=True)
            print(classification_report(actual, pred, digits=5))
        result["accuracy"] = result_to_print["accuracy"]
        result["Macro-F1"] = result_to_print["macro avg"]["f1-score"]
        result["Weighted-Macro-F1"] = result_to_print["weighted avg"]["f1-score"]
        return result
        
    def evaluate(self):
        accumulated_loss = 0.0
        accumulated_seq_cls_loss = 0.0
        accumulated_mul_cls_loss = 0.0
        accumulated_iit_cls_loss = 0.0
        
        accumulated_seq_cls_acc = 1.0/5
        accumulated_mul_cls_acc = 1.0/3
        accumulated_iit_cls_acc = 1.0/5
        
        # averaged.
        averaged_loss = 0.0
        averaged_seq_cls_loss = 0.0
        averaged_mul_cls_loss = 0.0
        averaged_iit_cls_loss = 0.0
        
        averaged_seq_cls_acc = 1.0/5
        averaged_mul_cls_acc = 1.0/3
        averaged_iit_cls_acc = 1.0/5
        
        n_sequences_epoch = 0
        n_effective_iit_sequence_epoch = 0
        n_effective_aspect_sequence_epoch = 0
        
        # labels saved for evaluation metrics.
        eval_predicted_labels = []
        eval_actual_labels = []
        
        self.high_level_model.model.eval()
        self.low_level_model.model.eval()
        with torch.no_grad():
            eval_dataloader = self.get_eval_dataloader()
            iter_bar = tqdm(eval_dataloader, desc="-Iter", disable=False)
            for batch in iter_bar:
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
                
                base_input_ids, base_attention_mask, \
                    base_labels, base_aspect_labels, \
                    source_input_ids, source_attention_mask, \
                    source_labels, source_aspect_labels = \
                self.prepare_batch(
                    *(input_ids, attention_mask, labels, aspect_labels)
                )
                
                base_intervention_mask, source_intervention_mask, \
                    base_intervention_corr, source_intervention_corr = self._get_interchange_mask(
                    base_aspect_labels, source_aspect_labels
                )

                # actual counterfactual labels with the high level model.
                _, _, counterfactual_labels = self.high_level_model.forward(
                    base_aspect_labels=base_aspect_labels,
                    source_aspect_labels=source_aspect_labels,
                    base_intervention_mask=base_intervention_mask,
                    source_intervention_mask=source_intervention_mask,
                )
                
                # send all data to gpus.
                base_labels = base_labels.to(self.device)
                source_labels = source_labels.to(self.device)
                counterfactual_labels = counterfactual_labels.to(self.device)
                base_aspect_labels = base_aspect_labels.to(self.device)
                source_aspect_labels = source_aspect_labels.to(self.device)
                base_input_ids = base_input_ids.to(self.device)
                base_attention_mask = base_attention_mask.to(self.device)
                source_input_ids = source_input_ids.to(self.device)
                source_attention_mask = source_attention_mask.to(self.device)
                base_intervention_corr = base_intervention_corr.to(self.device)
                source_intervention_corr = source_intervention_corr.to(self.device)

                # predicted counterfactual labels with the low level model.
                base_outputs, _, counterfactual_outputs = self.low_level_model.forward(
                    base=(base_input_ids, base_attention_mask),
                    source=(source_input_ids, source_attention_mask),
                    base_intervention_corr=base_intervention_corr,
                    source_intervention_corr=source_intervention_corr,
                )

                # various losses.
                seq_cls_loss, seq_cls_acc = self._seq_classification_loss(
                    base_outputs["logits"][0], base_labels.long()
                )

                (mul_cls_loss_0, mul_cls_loss_1, mul_cls_loss_2, mul_cls_loss_3), \
                    (mul_cls_acc_0, mul_cls_acc_1, mul_cls_acc_2, mul_cls_acc_3) = \
                        self._mul_classification_loss(*base_outputs["logits"][1:], base_aspect_labels.long())
                mul_cls_loss = (mul_cls_loss_0 + mul_cls_loss_1 + mul_cls_loss_2 + mul_cls_loss_3) / 4.0
                mul_cls_acc = (mul_cls_acc_0 + mul_cls_acc_1 + mul_cls_acc_2 + mul_cls_acc_3) / 4.0

                iit_cls_loss, iit_cls_acc = self._seq_classification_loss(
                    counterfactual_outputs["logits"][0], counterfactual_labels.long(), 
                    base_intervention_corr!=-1
                )

                loss = seq_cls_loss + self.alpha * mul_cls_loss + self.beta * iit_cls_loss
                if self.args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.

                # get the accumulated stats for stable perf audits.
                n_sample = base_input_ids.shape[0]
                accumulated_loss += loss * n_sample
                accumulated_seq_cls_loss += seq_cls_loss * n_sample
                accumulated_mul_cls_loss += mul_cls_loss * int((base_aspect_labels.long()!=-1).sum())
                accumulated_iit_cls_loss += iit_cls_loss * int((base_intervention_corr!=-1).sum())
                accumulated_seq_cls_acc += seq_cls_acc * n_sample
                accumulated_mul_cls_acc += mul_cls_acc * int((base_aspect_labels.long()!=-1).sum())
                accumulated_iit_cls_acc += iit_cls_acc * int((base_intervention_corr!=-1).sum())
                n_sequences_epoch += n_sample
                n_effective_aspect_sequence_epoch += int((base_aspect_labels.long()!=-1).sum())
                n_effective_iit_sequence_epoch += int((base_intervention_corr!=-1).sum())
        
                # save the label for metrics evaluation.
                _logits = base_outputs["logits"][0]
                loss_mask = torch.ones(_logits.shape[0]).bool().to(self.device)
                eval_predicted_labels.extend(_logits[loss_mask].data.max(1)[1].tolist())
                eval_actual_labels.extend(base_labels.long().tolist())

        # get the averaged stats for stable perf audits.
        eval_loss = accumulated_loss / n_sequences_epoch
        eval_seq_cls_loss = accumulated_seq_cls_loss / n_sequences_epoch
        eval_mul_cls_loss = accumulated_mul_cls_loss / n_effective_aspect_sequence_epoch
        eval_iit_cls_loss = accumulated_iit_cls_loss / n_effective_iit_sequence_epoch
        eval_seq_cls_acc = accumulated_seq_cls_acc / n_sequences_epoch
        eval_mul_cls_acc = accumulated_mul_cls_acc / n_effective_aspect_sequence_epoch
        eval_iit_cls_acc = accumulated_iit_cls_acc / n_effective_iit_sequence_epoch
        
        # metrics.
        seq_cls_eval_metrics = self._calculate_metrics(
            eval_actual_labels, eval_predicted_labels
        )
        
        # log eval results.
        log_eval = open(os.path.join(self.args.output_dir, 'eval_log.txt'), 'a', buffering=1)
        print('{},{},{},{},{},{},{},{}'.format(
                self.epoch+1,
                eval_loss,
                eval_seq_cls_loss, 
                eval_mul_cls_loss,
                eval_iit_cls_loss, 
                eval_seq_cls_acc,
                eval_mul_cls_acc,
                eval_iit_cls_acc,
            ),
            file=log_eval
        )
        log_eval.close()

        if "wandb" in self.args.report_to:
            wandb.log(
                {
                    "eval/loss": eval_loss, 
                    "eval/seq_cls_loss": eval_seq_cls_loss, 
                    "eval/mul_cls_loss": eval_mul_cls_loss, 
                    "eval/iit_cls_loss": eval_iit_cls_loss, 
                    "eval/seq_cls_acc": seq_cls_eval_metrics["accuracy"], # using this!
                    "eval/mul_cls_acc": eval_mul_cls_acc, 
                    "eval/iit_cls_acc": eval_iit_cls_acc, 
                    "eval/accuracy" : seq_cls_eval_metrics["accuracy"],
                    "eval/Macro-F1" : seq_cls_eval_metrics["Macro-F1"],
                    "eval/Weighted-Macro-F1" : seq_cls_eval_metrics["Weighted-Macro-F1"],
                    'epoch': self.epoch+1
                }, 
            )
        elif "none" in self.args.report_to:
            pass

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
        self.save_checkpoint()
        logger.info("Training is finished")
    
    def _get_interchange_mask(
        self,
        base_aspect_labels,
        source_aspect_labels,
    ):
        intervention_mask = torch.zeros_like(base_aspect_labels).bool()
        intervention_corr = []
        for i in range(0, base_aspect_labels.shape[0]):
            nonzero_indices = (
                (base_aspect_labels[i]!=-1)&(source_aspect_labels[i]!=-1)
            ).nonzero(as_tuple=False)
            if len(nonzero_indices) != 0:
                chosen_index = np.random.choice(nonzero_indices.flatten())
                intervention_corr += [chosen_index]
                intervention_mask[i, chosen_index] = True
            else:
                intervention_corr += [-1]
        return intervention_mask, intervention_mask, \
            torch.tensor(intervention_corr), torch.tensor(intervention_corr)
    
    def _seq_classification_loss(
        self,
        logits,
        labels,
        loss_mask=None,
    ):
        loss_mask = torch.ones(logits.shape[0]).bool().to(self.device) if loss_mask == None else loss_mask
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits[loss_mask], labels[loss_mask].view(-1))
        
        pred_cls = logits[loss_mask].data.max(1)[1]
        correct_count = pred_cls.eq(labels[loss_mask]).sum().cpu().item()
        
        return loss, (correct_count/logits.shape[0])*1.0
    
    def _mul_classification_loss(
        self,
        mul_logits_0, mul_logits_1, mul_logits_2, mul_logits_3,
        mul_labels,
    ):
        loss_0, acc_0 = self._seq_classification_loss(mul_logits_0, mul_labels[:,0], mul_labels[:,0]!=-1)
        loss_1, acc_1 = self._seq_classification_loss(mul_logits_1, mul_labels[:,1], mul_labels[:,1]!=-1)
        loss_2, acc_2 = self._seq_classification_loss(mul_logits_2, mul_labels[:,2], mul_labels[:,2]!=-1)
        loss_3, acc_3 = self._seq_classification_loss(mul_logits_3, mul_labels[:,3], mul_labels[:,3]!=-1)
        return (loss_0, loss_1, loss_2, loss_3), (acc_0, acc_1, acc_2, acc_3)
    
    def _record(
        self, n_sample, n_effective_aspect_sequence, n_effective_iit_sequence,
        loss, seq_cls_loss, mul_cls_loss, iit_cls_loss,
        seq_cls_acc, mul_cls_acc, iit_cls_acc,
    ):
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_seq_cls_loss = seq_cls_loss.mean().item() if self.args.n_gpu > 0 else seq_cls_loss.item()
        self.last_mul_cls_loss = mul_cls_loss.mean().item() if self.args.n_gpu > 0 else mul_cls_loss.item()
        self.last_iit_cls_loss = iit_cls_loss.mean().item() if self.args.n_gpu > 0 else iit_cls_loss.item()
        
        self.last_seq_cls_acc = seq_cls_acc
        self.last_mul_cls_acc = mul_cls_acc
        self.last_iit_cls_acc = iit_cls_acc
        
        # get the accumulated stats for stable perf audits.
        self.accumulated_loss += self.last_loss * n_sample
        self.accumulated_seq_cls_loss += self.last_seq_cls_loss * n_sample
        self.accumulated_mul_cls_loss += self.last_mul_cls_loss * n_effective_aspect_sequence
        self.accumulated_iit_cls_loss += self.last_iit_cls_loss * n_effective_iit_sequence
        self.accumulated_seq_cls_acc += self.last_seq_cls_acc * n_sample
        self.accumulated_mul_cls_acc +=self.last_mul_cls_acc * n_effective_aspect_sequence
        self.accumulated_iit_cls_acc += self.last_iit_cls_acc * n_effective_iit_sequence
        self.n_sequences_epoch += n_sample
        self.n_effective_aspect_sequence_epoch += n_effective_aspect_sequence
        self.n_effective_iit_sequence_epoch += n_effective_iit_sequence
        
        
        # get the averaged stats for stable perf audits.
        self.averaged_loss = self.accumulated_loss / self.n_sequences_epoch
        self.averaged_seq_cls_loss = self.accumulated_seq_cls_loss / self.n_sequences_epoch
        self.averaged_mul_cls_loss = self.accumulated_mul_cls_loss / self.n_effective_aspect_sequence_epoch
        self.averaged_iit_cls_loss = self.accumulated_iit_cls_loss / self.n_effective_iit_sequence_epoch
        self.averaged_seq_cls_acc = self.accumulated_seq_cls_acc / self.n_sequences_epoch
        self.averaged_mul_cls_acc = self.accumulated_mul_cls_acc / self.n_effective_aspect_sequence_epoch
        self.averaged_iit_cls_acc = self.accumulated_iit_cls_acc / self.n_effective_iit_sequence_epoch
    
    def step(
        self,
        base_input_ids, base_attention_mask, 
        base_labels, base_aspect_labels,
        source_input_ids, source_attention_mask, 
        source_labels, source_aspect_labels,
        skip_update_iter=False
    ):
        base_intervention_mask, source_intervention_mask, \
            base_intervention_corr, source_intervention_corr = self._get_interchange_mask(
            base_aspect_labels, source_aspect_labels
        )
        
        # actual counterfactual labels with the high level model.
        with torch.no_grad():
            _, _, counterfactual_labels = self.high_level_model.forward(
                base_aspect_labels=base_aspect_labels,
                source_aspect_labels=source_aspect_labels,
                base_intervention_mask=base_intervention_mask,
                source_intervention_mask=source_intervention_mask,
            )

        # send all data to gpus.
        base_labels = base_labels.to(self.device)
        source_labels = source_labels.to(self.device)
        counterfactual_labels = counterfactual_labels.to(self.device)
        base_aspect_labels = base_aspect_labels.to(self.device)
        source_aspect_labels = source_aspect_labels.to(self.device)
        base_input_ids = base_input_ids.to(self.device)
        base_attention_mask = base_attention_mask.to(self.device)
        source_input_ids = source_input_ids.to(self.device)
        source_attention_mask = source_attention_mask.to(self.device)
        base_intervention_corr = base_intervention_corr.to(self.device)
        source_intervention_corr = source_intervention_corr.to(self.device)
            
        # predicted counterfactual labels with the low level model.
        base_outputs, _, counterfactual_outputs = self.low_level_model.forward(
            base=(base_input_ids, base_attention_mask),
            source=(source_input_ids, source_attention_mask),
            base_intervention_corr=base_intervention_corr,
            source_intervention_corr=source_intervention_corr,
        )
        
        # various losses.
        seq_cls_loss, seq_cls_acc = self._seq_classification_loss(
            base_outputs["logits"][0], base_labels.long()
        )
        
        (mul_cls_loss_0, mul_cls_loss_1, mul_cls_loss_2, mul_cls_loss_3), \
            (mul_cls_acc_0, mul_cls_acc_1, mul_cls_acc_2, mul_cls_acc_3) = \
                self._mul_classification_loss(*base_outputs["logits"][1:], base_aspect_labels.long())
        mul_cls_loss = (mul_cls_loss_0 + mul_cls_loss_1 + mul_cls_loss_2 + mul_cls_loss_3) / 4.0
        mul_cls_acc = (mul_cls_acc_0 + mul_cls_acc_1 + mul_cls_acc_2 + mul_cls_acc_3) / 4.0
        
        iit_cls_loss, iit_cls_acc = self._seq_classification_loss(
            counterfactual_outputs["logits"][0], counterfactual_labels.long(), 
            base_intervention_corr!=-1
        )
        
        loss = seq_cls_loss + self.alpha * mul_cls_loss + self.beta * iit_cls_loss
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        
        self._record(
            base_input_ids.shape[0], 
            int((base_aspect_labels.long()!=-1).sum()), 
            int((base_intervention_corr!=-1).sum()),
            loss, seq_cls_loss, 
            mul_cls_loss, iit_cls_loss, seq_cls_acc, mul_cls_acc, iit_cls_acc
        )
        
        self.optimize(loss, skip_update_iter=skip_update_iter)
    
    def optimize(self, loss, skip_update_iter=False):
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        # backward()
        if self.args.fp16:
            assert False
        else:
            loss.backward()
        
        if not skip_update_iter:
            self.iter()

            if self.n_iter % self.args.gradient_accumulation_steps == 0:
                self.lr_this_step = self.optimizer.param_groups[0]['lr']
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
    
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
    
    def end_epoch(self):
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")
        
        if self.args.do_eval:
            self.evaluate()

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
        
        self.accumulated_seq_cls_acc = 1.0/5
        self.accumulated_mul_cls_acc = 1.0/3
        self.accumulated_iit_cls_acc = 1.0/5
    
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

        self._signature_columns = [
            "label", "label_ids", "input_ids", "attention_mask",
            'ambiance_label', 'food_label', 'noise_label', 'service_label'
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
    