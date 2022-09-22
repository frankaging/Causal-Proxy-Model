"""
In each of the model files, we have 2 types of models:
    - Basic NN
    - CPM Explainer Model
"""

from libs import *
from cpm import *

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

    
class CausalProxyModelForGPT2(CausalExplainer):
    def __init__(
        self, 
        blackbox_model_path,
        cpm_model_path, 
        device, batch_size, 
        intervention_h_dim=1,
        self_explain=False,
        random_source=False,
        probe_sample_source=False,
        cache_dir="../../huggingface_cache",
    ):
        self.batch_size = batch_size
        self.device = device
        self.self_explain = self_explain
        self.random_source = random_source
        self.probe_sample_source = probe_sample_source
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
        
        if "42" in cpm_model_path:
            self.seed = 42
        elif "66" in cpm_model_path:
            self.seed = 66
        elif "77" in cpm_model_path:
            self.seed = 77
    