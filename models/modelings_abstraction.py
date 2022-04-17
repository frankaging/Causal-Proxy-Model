import torch
import torch.nn as nn

class AbstractionModelForABSA(nn.Module):
    def __init__(
        self,
        model_type,
    ):
        super().__init__()
        self.num_aspect = 4
        self.num_dim_per_aspect = 3
        self.output_dim = 5
        self.dense = nn.Linear(
            self.num_aspect * self.num_dim_per_aspect, 
            self.num_aspect * self.num_dim_per_aspect
        )
        classifier_dropout = 0.0
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(
            self.num_aspect * self.num_dim_per_aspect, 
            self.output_dim
        )

        self.model_type = model_type
        
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
    ):
        if self.model_type == "majority_voting":
            neg_count = (aspect_labels==0).sum(dim=-1)
            pos_count = (aspect_labels==1).sum(dim=-1)
            unk_count = (aspect_labels==2).sum(dim=-1)
            
            very_neg_mask = (neg_count>pos_count+1)&(neg_count>pos_count)
            neg_mask = (neg_count==pos_count+1)
            neu_mask = (neg_count==pos_count)
            pos_mask = (pos_count==neg_count+1)
            very_pos_mask = (pos_count>neg_count+1)&(pos_count>neg_count)

            ver_neg_labels = torch.ones(aspect_labels.shape[0], device=aspect_labels.device)*0
            neg_labels = torch.ones(aspect_labels.shape[0], device=aspect_labels.device)*1
            neu_labels = torch.ones(aspect_labels.shape[0], device=aspect_labels.device)*2
            pos_labels = torch.ones(aspect_labels.shape[0], device=aspect_labels.device)*3
            very_pos_labels = torch.ones(aspect_labels.shape[0], device=aspect_labels.device)*4
            
            output_labels = very_neg_mask*ver_neg_labels \
                + neg_mask*neg_labels \
                + neu_mask*neu_labels \
                + pos_mask*pos_labels \
                + very_pos_mask*very_pos_labels
            
            return output_labels
            
        elif self.model_type == "logistic_regression":
            
            # [-1 0, 1, 2] -> [[0,0,0],[1,0,0],[0,1,0],[0,0,1]]
            aspect_0_mask = torch.zeros(aspect_labels.shape[0], 3, device=aspect_labels.device)
            aspect_0_mask[:,0] = (aspect_labels[:,0]==0)
            aspect_0_mask[:,1] = (aspect_labels[:,0]==1)
            aspect_0_mask[:,2] = (aspect_labels[:,0]==2)
            
            aspect_1_mask = torch.zeros(aspect_labels.shape[0], 3, device=aspect_labels.device)
            aspect_1_mask[:,0] = (aspect_labels[:,1]==0)
            aspect_1_mask[:,1] = (aspect_labels[:,1]==1)
            aspect_1_mask[:,2] = (aspect_labels[:,1]==2)
            
            aspect_2_mask = torch.zeros(aspect_labels.shape[0], 3, device=aspect_labels.device)
            aspect_2_mask[:,0] = (aspect_labels[:,2]==0)
            aspect_2_mask[:,1] = (aspect_labels[:,2]==1)
            aspect_2_mask[:,2] = (aspect_labels[:,2]==2)
            
            aspect_3_mask = torch.zeros(aspect_labels.shape[0], 3, device=aspect_labels.device)
            aspect_3_mask[:,0] = (aspect_labels[:,3]==0)
            aspect_3_mask[:,1] = (aspect_labels[:,3]==1)
            aspect_3_mask[:,2] = (aspect_labels[:,3]==2)

            binary_aspect_labels = torch.cat(
                [aspect_0_mask, aspect_1_mask, aspect_2_mask, aspect_3_mask], 
                dim=-1
            )
            
            x = binary_aspect_labels
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            output_logits = self.out_proj(x)
            
            return output_logits
            
        elif self.model_type == "human":
            raise NotImplementedError("To be implemented.")
        else:
            raise ValueError("Invalid tag.")
    
        return None

class InterventionableAbstractionModelForABSA():
    def __init__(
        self,
        model,
    ):
        self.model = model

    def forward(
        self,
        base_aspect_labels,
        source_aspect_labels,
        base_intervention_mask,
        source_intervention_mask,
        base_labels=None,
        source_labels=None,
    ):
        pred_base_labels = self.model(base_aspect_labels)
        pred_source_labels = self.model(source_aspect_labels)
        
        counterfactual_aspect_labels = base_aspect_labels.clone()
        counterfactual_aspect_labels[base_intervention_mask] = \
            source_aspect_labels[source_intervention_mask]
        counterfactual_labels = self.model(counterfactual_aspect_labels)
        
        return pred_base_labels, pred_source_labels, counterfactual_labels
        