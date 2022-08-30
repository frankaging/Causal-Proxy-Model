"""
In each of the model files, we have 2 types of models:
    - Basic NN
    - CPM Explainer Model
"""

from libs import *

class GPT2ForCEBaB(Model):
    def __init__(self, model_path, device='cpu', batch_size=64):
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = model_path
        self.batch_size = batch_size

        self.model = GPT2ForNonlinearSequenceClassification.from_pretrained(
            self.model_path,
            cache_dir="../../huggingface_cache"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT2 was trained without pad token but this is needed to batchify the data.
        # Because of the attention mask, the choice of pad_token will have no effect.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
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
        probas = np.round(probas.numpy(), decimals=4)

        predictions = np.argmax(probas, axis=1)
        clf_report = classification_report(y.to_numpy(), predictions, output_dict=True)

        return probas, clf_report

    def get_embeddings(self, sentences_list):
        x = self.tokenizer(sentences_list, padding=True, truncation=True, return_tensors='pt')
        embeddings = []
        for i in range(ceil(len(x['input_ids']) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x.items()}
            embeddings.append(self.model.transformer(**x_batch).last_hidden_state[:, 0, :].detach().cpu().tolist())

        return embeddings

    def get_classification_head(self):
        return self.model.score

    
class CausalProxyModelForGPT2(Explainer, CausalExplainer):
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
        self.blackbox_model = GPT2ForNonlinearSequenceClassification.from_pretrained(
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
        cpm_model = IITGPT2ForSequenceClassification.from_pretrained(
            cpm_model_path,
            config=cpm_config,
            cache_dir=cache_dir
        )
        cpm_model.to(device)
        self.cpm_model = InterventionableIITTransformerForSequenceClassification(
            model=cpm_model
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # GPT2 was trained without pad token but this is needed to batchify the data.
        # Because of the attention mask, the choice of pad_token will have no effect.
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.blackbox_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.cpm_model.model.config.pad_token_id = self.tokenizer.pad_token_id
        
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # we don't need to train IIT here.
        pass
    
    def preprocess(self, pairs_dataset, dev_dataset):
        
        # configs
        min_iit_pair_examples = self.min_iit_pair_examples
        match_non_int_type = self.match_non_int_type
        
        query_dataset = get_iit_examples(dev_dataset)
        iit_pairs_dataset = []
        iit_id = 0
        for index, row in pairs_dataset.iterrows():
            query_description_base = row['description_base']
            query_int_type = row['intervention_type']
            query_non_int_type = {
                "ambiance", "food", "noise", "service"
            } - {query_int_type}
            query_int_aspect_base = row["intervention_aspect_base"]
            query_int_aspect_assignment = row['intervention_aspect_counterfactual']
            query_original_id = row["original_id_base"]
            matched_iit_examples = query_dataset[
                (query_dataset[f"{query_int_type}_aspect_majority"]==query_int_aspect_assignment)&
                (query_dataset["original_id"]!=query_original_id)
            ]
            if match_non_int_type:
                for _t in query_non_int_type:
                    matched_iit_examples = matched_iit_examples[
                        (matched_iit_examples[f"{_t}_aspect_majority"]==\
                         row[f"{_t}_aspect_majority_base"])
                    ]
            if len(set(matched_iit_examples["id"])) < min_iit_pair_examples:
                if match_non_int_type:
                    # simply avoid mapping the rest of the aspects.
                    matched_iit_examples = query_dataset[
                        (query_dataset[f"{query_int_type}_aspect_majority"]==query_int_aspect_assignment)&
                        (query_dataset["original_id"]!=query_original_id)
                    ]
                else:
                    assert False # we need to check the number!
            sampled_iit_example_ids = random.sample(
                set(matched_iit_examples["id"]), min_iit_pair_examples
            )
            for _id in sampled_iit_example_ids:
                description_iit = query_dataset[query_dataset["id"]==_id]["description"].iloc[0]
                iit_pairs_dataset += [[
                    iit_id,
                    query_int_type,
                    query_description_base, 
                    description_iit
                ]]
            iit_id += 1
        iit_pairs_dataset = pd.DataFrame(
            columns=[
                'iit_id',
                'intervention_type', 
                'description_base', 
                'description_iit'], 
            data=iit_pairs_dataset
        )
        
        base_x = self.tokenizer(
            iit_pairs_dataset['description_base'].to_list(), 
            padding=True, truncation=True, return_tensors='pt'
        )
        source_x = self.tokenizer(
            iit_pairs_dataset['description_iit'].to_list(), 
            padding=True, truncation=True, return_tensors='pt'
        )
        intervention_corr = []
        for _type in iit_pairs_dataset["intervention_type"].tolist():
            if _type == "ambiance":
                intervention_corr += [0]
            if _type == "food":
                intervention_corr += [1]
            if _type == "noise":
                intervention_corr += [2]
            if _type == "service":
                intervention_corr += [3]
        intervention_corr = torch.tensor(intervention_corr).long()
        return base_x, source_x, intervention_corr, iit_pairs_dataset
    
    def estimate_icace(self, pairs, df):
        CPM_iTEs = []
        self.blackbox_model.eval()
        self.cpm_model.model.eval()
        base_x, source_x, intervention_corr, iit_pairs_dataset = self.preprocess(
            pairs, df
        )
        with torch.no_grad():
            for i in tqdm(range(ceil(len(iit_pairs_dataset)/self.batch_size))):
                base_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size].to(self.device) for k,v in base_x.items()} 
                source_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size].to(self.device) for k,v in source_x.items()} 
                intervention_corr_batch = intervention_corr[i*self.batch_size:(i+1)*self.batch_size].to(self.device)
                
                base_outputs = torch.nn.functional.softmax(
                    self.blackbox_model(**base_x_batch).logits.cpu(), dim=-1
                ).detach()
                _, _, counterfactual_outputs = self.cpm_model.forward(
                    base=(base_x_batch['input_ids'], base_x_batch['attention_mask']),
                    source=(source_x_batch['input_ids'], source_x_batch['attention_mask']),
                    base_intervention_corr=intervention_corr_batch,
                    source_intervention_corr=intervention_corr_batch,
                )
                counterfactual_outputs = torch.nn.functional.softmax(
                    counterfactual_outputs["logits"][0].cpu(), dim=-1
                ).detach()
                CPM_iTE = counterfactual_outputs-base_outputs
                CPM_iTEs.append(CPM_iTE)
        CPM_iTEs = torch.concat(CPM_iTEs)
        CPM_iTEs = np.round(CPM_iTEs.numpy(), decimals=4)

        # only for iit explainer!
        iit_pairs_dataset["EiCaCE"] = list(CPM_iTEs)
        CPM_iTEs = list(iit_pairs_dataset.groupby(["iit_id"])["EiCaCE"].mean())
        
        return CPM_iTEs