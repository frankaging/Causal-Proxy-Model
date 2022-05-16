import numpy as np
import torch
from eval_pipeline.explainers.abstract_explainer import Explainer
from eval_pipeline.utils import TREATMENTS, OVERALL_LABEL, DESCRIPTION, POSITIVE, NEGATIVE, UNKNOWN
from eval_pipeline.utils import unpack_batches
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from copy import deepcopy


class TCAV(Explainer):
    def __init__(self, original_model, treatments=TREATMENTS, device='cpu', batch_size=64):
        self.model = original_model
        self.treatments = treatments
        self.device = device
        self.batch_size = batch_size
        self.cavs = {}
        self.num_classes = 2

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        for treatment in self.treatments:
            preprocessed_dataset = self.train_preprocess(dataset, treatment)
            self.cavs[treatment] = self.learn_cav(unpack_batches(preprocessed_dataset[0]), treatment,
                                                  preprocessed_dataset[1])

    def predict_proba(self, pairs):
        scores = []
        embeddings, intervention_types = self.test_preprocess(pairs)
        clf_head = self.model.model.classifier.to(self.device)
        for idx, embedding in enumerate(embeddings):
            cav = self.cavs[intervention_types.iloc[idx]]
            grad = self.get_gradient(clf_head,
                                     torch.tensor(embedding, requires_grad=True, device=self.device, dtype=torch.float))
            classes_score = []
            for i in range(self.num_classes):
                classes_score.append((cav @ grad[i]))
            scores.append(np.array(classes_score))
        return scores

    def get_gradient(self, classifier, embedding):
        embedding.retain_grad()
        classifier.train()
        grads = []
        for k in range(self.num_classes):
            outputs = classifier(embedding)
            classifier.zero_grad()
            outputs[k].backward()
            grads.append(embedding.grad.detach().cpu().numpy())

        return grads

    def train_preprocess(self, dataset, treatment):
        treatment_labels = dataset[f'{treatment}_aspect_majority'].map({POSITIVE or NEGATIVE: 1, UNKNOWN: 0}).dropna()
        description = dataset[DESCRIPTION][treatment_labels.index].to_list()
        overall_labels = dataset[OVERALL_LABEL][treatment_labels.index].tolist()
        return self.model.get_embeddings(description), treatment_labels.tolist(), overall_labels

    def learn_cav(self, embeddings, treatment, cav_labels):
        # TODO set seed
        if len(set(cav_labels)) > 2:
            raise NotImplementedError('CAVs are binary by definition')

        # learn cav
        lm = linear_model.SGDClassifier(alpha=.01, max_iter=1_000, tol=1e-3)
        lm.fit(embeddings, cav_labels)
        accuracy = accuracy_score(cav_labels, lm.predict(embeddings))
        print(f'{treatment} cav accuracy --> {accuracy}')
        # format cav
        cav = -1 * lm.coef_[0]  # In binary classification the concept is assigned to label 0 by default, so flip coef_.
        cav = cav / np.linalg.norm(cav)  # normalize to unit vector

        return cav

    def test_preprocess(self, df):
        x = np.array(unpack_batches(self.model.get_embeddings(df['description_base'].tolist())))
        intervention_types = df['intervention_type']

        return x, intervention_types
