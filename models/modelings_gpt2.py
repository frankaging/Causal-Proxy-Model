import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import GPT2PreTrainedModel, GPT2Model
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

class GPT2NonlinearClassificationHead(nn.Module):
    """Head for sentence-level classification tasks. Identical to RobertaClassificationHead."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.n_embd, config.n_embd)
        classifier_dropout = (
            config.summary_first_dropout # 0.1
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.n_embd, config.num_labels)

    def forward(self, features, **kwargs):
        x = features  # features is the pooled [CLS] token
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class MultiTaskClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, num_labels):
        super().__init__()
        hidden_size = config.intervention_h_dim
        self.dense = nn.Linear(hidden_size, hidden_size)
        classifier_dropout = (
            config.summary_first_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class IITGPT2ForSequenceClassification(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.masked_bias", r"lm_head\.weight"]

    def __init__(
        self, config,
        num_aspect_labels=3,
    ):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = GPT2NonlinearClassificationHead(config)
        self.intervention_h_dim = config.intervention_h_dim
        self.multitask_classifier = MultiTaskClassificationHead(
            config, num_aspect_labels
        )
        
        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        # counterfactual arguments
        source_hidden_states=None,
        base_intervention_corr=None,
        source_intervention_corr=None,
        all_layers=None,
        cls_hidden_reprs=None,
        # counterfactual arguments
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]
        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )
                
        if cls_hidden_reprs is not None:
            # we also need to all the pooler once if configured.
            pooled_hidden_states = cls_hidden_reprs
        else:
            transformer_outputs = self.transformer(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            hidden_states = transformer_outputs[0]
            pooled_hidden_states = hidden_states[torch.arange(batch_size, device=self.device), sequence_lengths]
        #####
        # We can simply do IIT here as well as the multitask objective!
        # INT_POINT: only the last layer!
        if base_intervention_corr is not None and source_hidden_states is not None:
            for b in range(0, pooled_hidden_states.shape[0]):
                if base_intervention_corr[b] != -1:
                    start_idx = base_intervention_corr[b]*self.intervention_h_dim
                    end_idx = (base_intervention_corr[b]+1)*self.intervention_h_dim
                    # we support where the pass in source_hidden_states
                    # is a partial one only for the interchanging aspect.
                    if not isinstance(source_hidden_states, tuple) and pooled_hidden_states.shape[-1] != source_hidden_states.shape[-1]:
                        pooled_hidden_states[b][start_idx:end_idx] = source_hidden_states[b]
                    else:
                        pooled_hidden_states[b][start_idx:end_idx] = \
                            source_hidden_states[b][start_idx:end_idx]
                        
        mul_logits_0 = self.multitask_classifier(pooled_hidden_states[:,:self.intervention_h_dim])
        mul_logits_1 = self.multitask_classifier(pooled_hidden_states[:,self.intervention_h_dim:self.intervention_h_dim*2])
        mul_logits_2 = self.multitask_classifier(pooled_hidden_states[:,self.intervention_h_dim*2:self.intervention_h_dim*3])
        mul_logits_3 = self.multitask_classifier(pooled_hidden_states[:,self.intervention_h_dim*3:self.intervention_h_dim*4])
        #####
        
        logits = self.score(pooled_hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=(logits, mul_logits_0, mul_logits_1, mul_logits_2, mul_logits_3),
            past_key_values=None if cls_hidden_reprs is not None else transformer_outputs.past_key_values,
            hidden_states=None if cls_hidden_reprs is not None else pooled_hidden_states,
            attentions=None if cls_hidden_reprs is not None else transformer_outputs.attentions,
        )