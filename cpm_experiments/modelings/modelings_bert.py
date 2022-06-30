from libs import *

class BERTForCEBaB(Model):
    def __init__(self, model_path, device='cpu', batch_size=64):
        self.device = device
        self.model_path = model_path
        self.tokenizer_path = model_path
        self.batch_size = batch_size

        self.model = BertForNonlinearSequenceClassification.from_pretrained(
            self.model_path,
            cache_dir="../../huggingface_cache"
        )
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

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

class CausalExplainer(object):
    def __init__(self):
        pass

class CausalMediationModelForBERT(Explainer, CausalExplainer):
    def __init__(
        self, 
        model_path, 
        device, batch_size, 
        intervention_h_dim=1,
        min_iit_pair_examples=1,
        match_non_int_type=False,
        cache_dir="../../huggingface_cache",
        align_neurons=None,
    ):
        self.batch_size = batch_size
        self.device = device
        self.min_iit_pair_examples = min_iit_pair_examples
        self.match_non_int_type = match_non_int_type
        
        # causal proxy model loading.
        model_config = AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            use_auth_token=True if "CEBaB/" in model_path else False,
        )
        try:
            model_config.intervention_h_dim = model_config.intervention_h_dim
        except:
            model_config.intervention_h_dim = intervention_h_dim
        self.intervention_h_dim = intervention_h_dim
        model = IITBERTForSequenceClassification.from_pretrained(
            model_path,
            config=model_config,
            cache_dir=cache_dir
        )
        model.to(device)
        self.model = InterventionableIITTransformerForSequenceClassification(
            model=model
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.align_neurons = align_neurons
        
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # in case, the align_neurons is not provided, we need to re-fit.
        if self.align_neurons:
            return
    
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
        # TODO: dummy code here.
        intervention_type = []
        for _type in iit_pairs_dataset["intervention_type"].tolist():
            if _type == "ambiance":
                intervention_type += ["ambiance"]
            if _type == "food":
                intervention_type += ["food"]
            if _type == "noise":
                intervention_type += ["noise"]
            if _type == "service":
                intervention_type += ["service"]
        return base_x, source_x, intervention_type, iit_pairs_dataset
    
    def estimate_icace(self, pairs, df):
        CPM_iTEs = []
        self.model.model.eval()
        base_x, source_x, intervention_type, iit_pairs_dataset = self.preprocess(
            pairs, df
        )
        with torch.no_grad():
            for i in tqdm(range(ceil(len(iit_pairs_dataset)/self.batch_size))):
                base_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size].to(self.device) for k,v in base_x.items()} 
                source_x_batch = {k:v[i*self.batch_size:(i+1)*self.batch_size].to(self.device) for k,v in source_x.items()} 
                intervention_type_batch = intervention_type[i*self.batch_size:(i+1)*self.batch_size]
                # base output.
                outputs = self.model.model(
                    **base_x_batch,
                    output_hidden_states=True,
                )
                base_logits = torch.nn.functional.softmax(
                    outputs.logits[0].cpu(), dim=-1
                ).detach()
                base_cls_hidden_state = outputs.hidden_states[-1][:,0,:].detach()
                # source output.
                source_outputs = self.model.model(
                    **source_x_batch,
                    output_hidden_states=True,
                )
                source_cls_hidden_state = source_outputs.hidden_states[-1][:,0,:].detach()
                # intervention.
                selected_neurons = self.align_neurons[intervention_type_batch[0]]
                for neuron_id in selected_neurons:
                    base_cls_hidden_state[:, neuron_id] = source_cls_hidden_state[:, neuron_id]
                # counterfactual output.
                counterfactual_outputs, _, _ = self.model.forward_with_cls_hidden_reprs(
                    cls_hidden_reprs=base_cls_hidden_state.unsqueeze(dim=1)
                )
                counterfactual_logits = torch.nn.functional.softmax(
                        counterfactual_outputs.logits[0].cpu(), dim=-1
                ).detach()[0]
                # estimate the effect.
                CPM_iTE = counterfactual_logits-base_logits
                CPM_iTEs.append(CPM_iTE)
        CPM_iTEs = torch.concat(CPM_iTEs)
        CPM_iTEs = np.round(CPM_iTEs.numpy(), decimals=4)

        # only for iit explainer!
        iit_pairs_dataset["EiCaCE"] = list(CPM_iTEs)
        CPM_iTEs = list(iit_pairs_dataset.groupby(["iit_id"])["EiCaCE"].mean())
        
        return CPM_iTEs