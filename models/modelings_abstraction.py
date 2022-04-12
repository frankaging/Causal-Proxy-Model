import torch
import torch.nn as nn

class AbstractionModelForABSA(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.aspect_sequence = [
            'ambiance_label', 'food_label', 'noise_label', 'service_label'
        ]
    """
    0: neg
    1: pos
    2: unknown
    
    type:
        - majority_voting
            - (0, 0, 0, 0) -> very negative
            - (0, 0, 0, 1) -> negative
            - (0, 0, 1, 1) -> neutral
            - (0, 1, 1, 1) -> positive
            - (1, 1, 1, 1) -> very positive
            == special cases ==
            - (2, 2, 2, 2) -> neutral
            - (2, 1, 1, 1) -> very positive
            - (2, 2, 1, 1) -> positive
            - (2, 2, 2, 1) -> positive
            - (2, 0, 0, 0) -> very negative
            - (2, 2, 0, 0) -> negative
            - (2, 2, 2, 0) -> negative
            - (2, 2, 1, 0) -> neutral
            - (2, 1, 1, 0) -> positive
            - (2, 1, 0, 0) -> negative
    """
    def forward(
        self, aspect_labels,
        tag="majority_voting",
    ):
        if tag == "majority_voting":
            neg_count = (aspect_labels==0).sum(dim=-1)
            pos_count = (aspect_labels==1).sum(dim=-1)
            unk_count = (aspect_labels==2).sum(dim=-1)
            
            very_neg_mask = (neg_count>pos_count+1)&(neg_count>pos_count)
            neg_mask = (neg_count==pos_count+1)
            neu_mask = (neg_count==pos_count)
            pos_mask = (pos_count==neg_count+1)
            very_pos_mask = (pos_count>neg_count+1)&(pos_count>neg_count)

            ver_neg_labels = torch.ones(aspect_labels.shape[0])*0
            neg_labels = torch.ones(aspect_labels.shape[0])*1
            neu_labels = torch.ones(aspect_labels.shape[0])*2
            pos_labels = torch.ones(aspect_labels.shape[0])*3
            very_pos_labels = torch.ones(aspect_labels.shape[0])*4
            
            output_labels = very_neg_mask*ver_neg_labels \
                + neg_mask*neg_labels \
                + neu_mask*neu_labels \
                + pos_mask*pos_labels \
                + very_pos_mask*very_pos_labels
            
        elif tag == "weighted_majority_voting":
            raise NotImplementedError("To be implemented.")
        elif tag == "human":
            raise NotImplementedError("To be implemented.")
        else:
            raise ValueError("Invalid tag.")
    
        return output_labels

class InterventionableAbstractionModelForABSA():
    def __init__(
        self,
        model,
    ):
        self.model = model
        self.aspect_sequence = [
            'ambiance_label', 'food_label', 'noise_label', 'service_label'
        ]

    def forward(
        self,
        base_aspect_labels,
        source_aspect_labels,
        base_intervention_mask,
        source_intervention_mask,
        tag="majority_voting",
        base_labels=None,
        source_labels=None,
    ):
        if base_labels == None:
            base_labels = self.model(base_aspect_labels)
        if source_labels == None:
            source_labels = self.model(source_aspect_labels)
        
        counterfactual_aspect_labels = base_aspect_labels
        counterfactual_aspect_labels[base_intervention_mask] = \
            source_aspect_labels[source_intervention_mask]
        counterfactual_labels = self.model(counterfactual_aspect_labels)
        
        return base_labels, source_labels, counterfactual_labels
        