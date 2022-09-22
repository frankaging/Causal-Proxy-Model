"""
In each of the model files, we have 3 types of models:
    - Basic NN
    - CPM Explainer Model
    - Causal Mediation Explainer Model 
"""

from libs import *

class CausalExplainer(Explainer):
    def __init__(self):
        pass

    def preprocess_predict_multitask_proba(self, df):
        x = self.tokenizer(df['text'].to_list(), padding=True, truncation=True, return_tensors='pt')
        ambiance_y = df['ambiance_label'].astype(int).tolist()
        food_y = df['food_label'].astype(int).tolist()
        noise_y = df['noise_label'].astype(int).tolist()
        service_y = df['service_label'].astype(int).tolist()

        return x, \
            torch.tensor(ambiance_y).long(), \
            torch.tensor(food_y).long(), \
            torch.tensor(noise_y).long(), \
            torch.tensor(service_y).long()
    
    def predict_multitask_proba(self, dataset):
        self.cpm_model.model.eval()

        x, ambiance_y, food_y, noise_y, service_y = \
            self.preprocess_predict_multitask_proba(dataset)
    
        # get the predictions batch per batch
        ambiance_probas = []
        food_probas = []
        noise_probas = []
        service_probas = []
        for i in range(ceil(len(dataset) / self.batch_size)):
            x_batch = {k: v[i * self.batch_size:(i + 1) * self.batch_size].to(self.device) for k, v in x.items()}
            outputs = self.cpm_model.model(**x_batch)
            ambiance_probas.append(
                torch.nn.functional.softmax(
                    outputs.logits[1].cpu(), dim=-1
                ).detach()
            )
            food_probas.append(
                torch.nn.functional.softmax(
                    outputs.logits[2].cpu(), dim=-1
                ).detach()
            )
            noise_probas.append(
                torch.nn.functional.softmax(
                    outputs.logits[3].cpu(), dim=-1
                ).detach()
            )
            service_probas.append(
                torch.nn.functional.softmax(
                    outputs.logits[4].cpu(), dim=-1
                ).detach()
            )

        ambiance_probas = torch.concat(ambiance_probas)
        food_probas = torch.concat(food_probas)
        noise_probas = torch.concat(noise_probas)
        service_probas = torch.concat(service_probas)
        
        ambiance_predictions = np.argmax(ambiance_probas, axis=1)
        food_predictions = np.argmax(food_probas, axis=1)
        noise_predictions = np.argmax(noise_probas, axis=1)
        service_predictions = np.argmax(service_probas, axis=1)
        
        ambiance_y_mask = ambiance_y!=-1
        food_y_mask = food_y!=-1
        noise_y_mask = noise_y!=-1
        service_y_mask = service_y!=-1
        
        ambiance_clf_report = classification_report(
            torch.masked_select(ambiance_y, ambiance_y_mask).numpy(), 
            torch.masked_select(ambiance_predictions, ambiance_y_mask),
            output_dict=True
        )
        food_clf_report = classification_report(
            torch.masked_select(food_y, food_y_mask).numpy(), 
            torch.masked_select(food_predictions, food_y_mask),
            output_dict=True
        )
        noise_clf_report = classification_report(
            torch.masked_select(noise_y, noise_y_mask).numpy(), 
            torch.masked_select(noise_predictions, noise_y_mask),
            output_dict=True
        )
        service_clf_report = classification_report(
            torch.masked_select(service_y, service_y_mask).numpy(), 
            torch.masked_select(service_predictions, service_y_mask),
            output_dict=True
        )
        print("ambiance acc=", ambiance_clf_report['accuracy'])
        print("food acc=", food_clf_report['accuracy'])
        print("noise acc=", noise_clf_report['accuracy'])
        print("service acc=", service_clf_report['accuracy'])

        return ambiance_predictions, food_predictions, noise_predictions, service_predictions, \
            ambiance_clf_report, food_clf_report, noise_clf_report, service_clf_report
    
    def refresh_concept_labels_with_probes(self, dev_dataset):
        print("refreshing your concept labels with probe-based predictions ...")
        dev_dataset_copy = dev_dataset.copy()
        aspect_label_encode = {
            "Negative":0,
            "Positive":1,
            "unknown":2,
            "no majority": 2,
        }

        columns_to_keep = [
            'original_id', 'edit_id', 'is_original', 
            'description', 'review_majority',
            'food_aspect_majority', 'ambiance_aspect_majority', 
            'service_aspect_majority', 'noise_aspect_majority',
        ]
        dev_dataset_copy = dev_dataset_copy[columns_to_keep]
        dev_dataset_copy = dev_dataset_copy.rename(
            columns={
                'description': 'text', 
                'review_majority': 'label',
                'food_aspect_majority': 'food_label',
                'ambiance_aspect_majority': 'ambiance_label',
                'service_aspect_majority': 'service_label',
                'noise_aspect_majority': 'noise_label'
            }
        )
        dev_dataset_copy = dev_dataset_copy.replace("", -1).replace(
            {
                "food_label": aspect_label_encode,
                "ambiance_label": aspect_label_encode,
                "service_label": aspect_label_encode,
                "noise_label": aspect_label_encode
            }
        )
        ambiance_predictions, food_predictions, noise_predictions, service_predictions, \
                ambiance_clf_report, food_clf_report, noise_clf_report, service_clf_report = \
                self.predict_multitask_proba(dev_dataset_copy)
        
        dev_dataset_return_copy = dev_dataset.copy()
        # replace with all predictions
        dev_dataset_return_copy["ambiance_aspect_majority"] = ambiance_predictions.tolist()
        dev_dataset_return_copy["food_aspect_majority"] = food_predictions.tolist()
        dev_dataset_return_copy["noise_aspect_majority"] = noise_predictions.tolist()
        dev_dataset_return_copy["service_aspect_majority"] = service_predictions.tolist()
        
        reverse_aspect_label_encode = {
            0 : "Negative",
            1 : "Positive",
            2 : "unknown",
        }
        dev_dataset_return_copy = dev_dataset_return_copy.replace(
            {
                "ambiance_aspect_majority": reverse_aspect_label_encode,
                "food_aspect_majority": reverse_aspect_label_encode,
                "noise_aspect_majority": reverse_aspect_label_encode,
                "service_aspect_majority": reverse_aspect_label_encode
            }
        )

        return dev_dataset_return_copy
    
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
            if self.random_source:
                satisfied_rows = dev_dataset # just enforce it to random sample from the training data
            else:
                satisfied_rows = dev_dataset[
                    (dev_dataset[f"{intervention_type}_aspect_majority"]==\
                     intervention_aspect_counterfactual)
                ]
                if len(satisfied_rows) == 0:
                    # this has some complications. our probes at some condition may not work well.
                    # in these cases, our probe may only predict certain values. we sample randomly.
                    satisfied_rows = dev_dataset
            sampled_source = satisfied_rows.sample(random_state=self.seed).iloc[0]
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
        if self.probe_sample_source:
            sample_df = self.refresh_concept_labels_with_probes(df)
        else:
            sample_df = df
        base_x, source_x, intervention_type, prediction_base = self.preprocess(
            pairs, sample_df
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
                
                if self.self_explain:
                    prediction_base_self_batch = torch.nn.functional.softmax(
                        self.cpm_model.model(
                            input_ids=base_input_ids,
                            attention_mask=base_attention_mask,
                        ).logits[0].cpu(), dim=-1
                    ).detach()
                    CPM_iTE = prediction_counterfactual_batch-prediction_base_self_batch
                else:
                    CPM_iTE = prediction_counterfactual_batch-prediction_base_batch
                
                CPM_iTEs.append(CPM_iTE)
        CPM_iTEs = torch.concat(CPM_iTEs)
        CPM_iTEs = CPM_iTEs.numpy()
        return list(CPM_iTEs)
    