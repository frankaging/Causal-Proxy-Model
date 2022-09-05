"""
In each of the model files, we have 2 types of models:
    - Basic NN
    - CPM Explainer Model
"""

from libs import *
from cpm import *

class RoBERTaForCEBaB(Model):
    def __init__(self, model_path, device='cpu', batch_size=64):
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = model_path
        self.batch_size = batch_size

        self.model = RobertaForNonlinearSequenceClassification.from_pretrained(
            self.model_path,
            cache_dir="../../huggingface_cache"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        self.model.to(device)

    def __str__(self):
        return self.model_path.split('/')[-1]

    def preprocess(self, df):
        x = self.tokenizer(df['description'].to_list(), padding=True, truncation=True, return_tensors='pt')
        y = df['review_majority'].astype(int)

        return x, y

    def fit(self, dataset):
        # assume model was already trained
        pass

    def predict_proba(self, dataset):
        self.model.eval()

        x, y = self.preprocess(dataset)

        # get the predictions batch per batch
        probas = []
        for i in range(ceil(len(dataset) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x.items()}
            probas.append(torch.nn.functional.softmax(self.model(**x_batch).logits.cpu(), dim=-1).detach())

        probas = torch.concat(probas)
        probas = np.round(probas.numpy(), decimals=16)

        predictions = np.argmax(probas, axis=1)
        clf_report = classification_report(y.to_numpy(), predictions, output_dict=True)

        return probas, clf_report

    def get_embeddings(self, sentences_list):
        x = self.tokenizer(sentences_list, padding=True, truncation=True, return_tensors='pt')
        embeddings = []
        for i in range(ceil(len(x['input_ids']) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x.items()}
            embeddings.append(self.model.base_model(**x_batch).pooler_output.detach().cpu().tolist())

        return embeddings

    def get_classification_head(self):
        return self.model.classifier

    
class CausalProxyModelForRoBERTa(Explainer, CausalExplainer):
    def __init__(
        self, 
        blackbox_model_path,
        cpm_model_path, 
        device, batch_size, 
        intervention_h_dim=1,
        min_iit_pair_examples=1,
        match_non_int_type=False,
        cache_dir="../../huggingface_cache",
    ):
        self.batch_size = batch_size
        self.device = device
        self.min_iit_pair_examples = min_iit_pair_examples
        self.match_non_int_type = match_non_int_type
        # blackbox model loading.
        self.blackbox_model = RobertaForNonlinearSequenceClassification.from_pretrained(
            blackbox_model_path,
            cache_dir=cache_dir
        )
        self.blackbox_model.to(device)
        
        # causal proxy model loading.
        cpm_config = AutoConfig.from_pretrained(
            cpm_model_path,
            cache_dir=cache_dir,
            use_auth_token=True if "CEBaB/" in cpm_model_path else False,
        )

        print(f"intervention_h_dim={cpm_config.intervention_h_dim}")
        cpm_model = IITRobertaForSequenceClassification.from_pretrained(
            cpm_model_path,
            config=cpm_config,
            cache_dir=cache_dir
        )
        cpm_model.to(device)
        self.cpm_model = InterventionableIITTransformerForSequenceClassification(
            model=cpm_model
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
    def preprocess_predict_proba(self, df):
        x = self.tokenizer(df['description'].to_list(), padding=True, truncation=True, return_tensors='pt')
        y = df['review_majority'].astype(int)

        return x, y
        
    def predict_proba(self, dataset):
        self.cpm_model.model.eval()

        x, y = self.preprocess_predict_proba(dataset)

        # get the predictions batch per batch
        probas = []
        for i in range(ceil(len(dataset) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x.items()}
            probas.append(torch.nn.functional.softmax(self.cpm_model.model(**x_batch).logits[0].cpu(), dim=-1).detach())
        probas = torch.concat(probas)

        predictions = np.argmax(probas, axis=1)
        clf_report = classification_report(y.to_numpy(), predictions, output_dict=True)

        return probas, clf_report
        
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # we don't need to train IIT here.
        pass
    
    def preprocess(self, pairs_dataset, dev_dataset):        
        iit_pairs_dataset = []
        for index, row in pairs_dataset.iterrows():
            description_base = row['description_base']
            prediction_base = row['prediction_base']
            intervention_type = row['intervention_type']
            intervention_aspect_counterfactual = row['intervention_aspect_counterfactual']
            satisfied_rows = dev_dataset[
                (dev_dataset[f"{intervention_type}_aspect_majority"]==\
                 intervention_aspect_counterfactual)
            ]
            sampled_source = satisfied_rows.sample().iloc[0]
            iit_pairs_dataset += [[
                intervention_type,
                row['description_base'],
                sampled_source["description"],
                row["prediction_base"],
            ]]

        iit_pairs_dataset = pd.DataFrame(
            columns=[
                'intervention_type',
                'description_base', 
                'description_counterfactual', 
                'prediction_base'], 
            data=iit_pairs_dataset
        )
        aspect_encode = {
            "ambiance":0,
            "food":1,
            "noise":2,
            "service": 3,
        }
        iit_pairs_dataset = iit_pairs_dataset.replace(
            {"intervention_type": aspect_encode,}
        )
        
        base_x = self.tokenizer(
            iit_pairs_dataset['description_base'].to_list(), 
            padding=True, truncation=True, return_tensors='pt'
        )
        source_x = self.tokenizer(
            iit_pairs_dataset['description_counterfactual'].to_list(), 
            padding=True, truncation=True, return_tensors='pt'
        )
        intervention_type = torch.tensor(iit_pairs_dataset["intervention_type"]).long()
        prediction_base = np.array(
            [np.array(arr) for arr in iit_pairs_dataset["prediction_base"]]
        )
        prediction_base = torch.tensor(prediction_base).float()
        return base_x, source_x, intervention_type, prediction_base
    
    def estimate_icace(self, pairs, df):
        CPM_iTEs = []
        self.cpm_model.model.eval()
        base_x, source_x, intervention_type, prediction_base = self.preprocess(
            pairs, df
        )
        with torch.no_grad():
            for i in tqdm(range(ceil(intervention_type.shape[0]/self.batch_size))):
                base_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size] for k,v in base_x.items()} 
                source_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size] for k,v in source_x.items()} 
                intervention_type_batch = intervention_type[i*self.batch_size:(i+1)*self.batch_size]
                prediction_base_batch = prediction_base[i*self.batch_size:(i+1)*self.batch_size]
                
                base_input_ids = base_x_batch['input_ids']
                base_attention_mask = base_x_batch['attention_mask']
                source_input_ids = source_x_batch['input_ids']
                source_attention_mask = source_x_batch['attention_mask']
                base_input_ids = base_input_ids.to(self.device)
                base_attention_mask = base_attention_mask.to(self.device)
                source_input_ids = source_input_ids.to(self.device)
                source_attention_mask = source_attention_mask.to(self.device)
                intervention_type_batch = intervention_type_batch.to(self.device)
                
                _, _, counterfactual_outputs = self.cpm_model.forward(
                    base=(base_input_ids, base_attention_mask),
                    source=(source_input_ids, source_attention_mask),
                    base_intervention_corr=intervention_type_batch,
                    source_intervention_corr=intervention_type_batch,
                )
                prediction_counterfactual_batch = torch.nn.functional.softmax(
                    counterfactual_outputs["logits"][0].cpu(), dim=-1
                ).detach()
                CPM_iTE = prediction_counterfactual_batch-prediction_base_batch
                CPM_iTEs.append(CPM_iTE)
        CPM_iTEs = torch.concat(CPM_iTEs)
        CPM_iTEs = CPM_iTEs.numpy()
        return list(CPM_iTEs)
    