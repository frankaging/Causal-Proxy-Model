import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import null_space
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from eval_pipeline.utils import OVERALL_LABEL, DESCRIPTION, TREATMENTS, POSITIVE, NEGATIVE, UNKNOWN
from eval_pipeline.utils import unpack_batches
from eval_pipeline.explainers.abstract_explainer import Explainer


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn


class INLP(Explainer):
    def __init__(self, figure_path=None, treatments=TREATMENTS, device='cpu', batch_size=64):
        self.device = device
        self.batch_size = batch_size
        self.projection_matrices, self.clfs, self.inlp_clfs = {}, {}, {}
        self.treatments = treatments
        self.model = None
        self.figure_path = figure_path

    @staticmethod
    def treatment_to_label(x):
        if x == NEGATIVE:
            return 0
        elif x == UNKNOWN:
            return 1
        elif x == POSITIVE:
            return 2
        else:
            return None

    def train_preprocess(self, dataset, treatment):
        # TODO fix data type
        treatment_labels = dataset[f'{treatment}_aspect_majority'].apply(self.treatment_to_label).dropna().astype(int)
        description = dataset[DESCRIPTION][treatment_labels.index].to_list()
        overall_labels = dataset[OVERALL_LABEL][treatment_labels.index].tolist()
        return self.model.get_embeddings(description), treatment_labels.tolist(), overall_labels

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset):
        # TODO should we add a dev set access to fit?
        self.model = classifier
        model_clf_head = self.model.model.classifier
        for treatment in self.treatments:
            preprocessed_dataset = self.train_preprocess(dataset, treatment)
            dev_preprocessed = self.train_preprocess(dev_dataset, treatment)
            embeddings = np.array(unpack_batches(preprocessed_dataset[0]))
            dev_embeddings = np.array(unpack_batches(dev_preprocessed[0]))
            self.projection_matrices[treatment] = self.inlp_method(embeddings, preprocessed_dataset[1], dev_embeddings,
                                                                   dev_preprocessed[1], treatment)
            inlp_embeddings = embeddings @ self.projection_matrices[treatment]
            overall_labels = np.array(preprocessed_dataset[2])
            dev_overall_labels = np.array(dev_preprocessed[2])
            self.clfs[treatment] = self.train_clf(embeddings, overall_labels, dev_embeddings, dev_overall_labels,
                                                  clf_model=deepcopy(model_clf_head),
                                                  clf_name=f'overall_task_{treatment}').float()
            self.inlp_clfs[treatment] = self.train_clf(inlp_embeddings, overall_labels, dev_embeddings,
                                                       dev_overall_labels,
                                                       clf_model=deepcopy(model_clf_head),
                                                       clf_name=f'inlp_overall_task_{treatment}').float()

    def predict_proba(self, pairs):
        probas_lst = []
        embeddings, intervention_types = self.test_preprocess(pairs)
        for idx, embedding in enumerate(embeddings):
            with torch.no_grad():
                inlp_clf = self.inlp_clfs[intervention_types.iloc[idx]]
                clf = self.clfs[intervention_types.iloc[idx]]
                probas = torch.softmax(clf(torch.tensor(embedding).to('cuda').float()), dim=0)
                inlp_probas = torch.softmax(inlp_clf(
                    torch.tensor(embedding @ self.projection_matrices[intervention_types.iloc[idx]]).to(
                        'cuda').float()), dim=0)
                probas_lst.append((inlp_probas - probas).cpu().numpy())
        return list(probas_lst)

    def inlp_method(self, X_train, y_train, X_dev, y_dev, treatment, random_seed=0, iterations=30):
        # TODO check in the paper details about "how to choose number of iterations"
        train_accuracies, dev_accuracies = [], []
        X_projected = X_train
        p_intersection = np.eye(X_projected.shape[1], X_projected.shape[1])
        for _ in np.arange(iterations):
            # TODO try with different linear model, get it from args
            clf = LogisticRegression(random_state=random_seed).fit(X_projected, y_train)
            w = clf.coef_
            preds_on_train = clf.predict(X_projected)
            train_accuracies.append(accuracy_score(preds_on_train, y_train))
            dev_accuracies.append(accuracy_score(clf.predict(X_dev @ p_intersection), y_dev))
            b = null_space(w)
            p_null_space = b @ b.T
            p_intersection = p_intersection @ p_null_space
            X_projected = (p_null_space @ X_projected.T).T
        plt.plot(train_accuracies, label='train'), plt.plot(dev_accuracies, label='dev')
        plt.title(f'probing {treatment} per iteration')
        plt.legend()
        if self.figure_path:
            plt.savefig(os.path.join(self.figure_path, treatment))
        plt.clf()
        return p_intersection

    def train_clf(self, X_train, y_train, X_dev, y_dev, clf_model, clf_name):
        print(f'starting training {clf_name}')

        # TODO get these numbers through args, BTW does the number of epochs make sense to Eldar?
        learning_rate = 2e-5
        num_epochs = 30
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(clf_model.parameters(), lr=learning_rate)
        # TODO tallk with Eldar about the choosing of the hyperparameters + scheduler?
        # scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)
        train_embeddings = torch.from_numpy(X_train).float().to('cuda')
        dev_embeddings = torch.from_numpy(X_dev).float().to('cuda')
        train_labels = torch.from_numpy(y_train).float().to('cuda')
        dev_labels = torch.from_numpy(y_dev).float().to('cuda')
        train_accuracies = []
        dev_accuracies = []
        for epoch in range(num_epochs):

            logits = clf_model(train_embeddings)
            loss = criterion(logits, train_labels.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(logits, 1)
            train_accuracy = (predicted == train_labels).sum() / len(train_labels)
            with torch.no_grad():
                dev_accuracy = (torch.argmax(clf_model(dev_embeddings), 1) == dev_labels).sum() / len(dev_labels)
            if epoch % 10 == 0:
                print(
                    f'{clf_name}- epoch: {epoch} loss: {loss:.3f} accuracy: {train_accuracy :.3f}, dev: {dev_accuracy :.3f}')
                # scheduler.step(train_accuracy)
            train_accuracies.append(train_accuracy.cpu())
            dev_accuracies.append(dev_accuracy.cpu())

        clf_model.eval()
        print(f'{clf_name}: training accuracy: {train_accuracy :.3f}')
        plt.plot(train_accuracies, label='train'), plt.plot(dev_accuracies, label='dev')
        plt.title(clf_name)
        plt.legend()
        if self.figure_path:
            plt.savefig(os.path.join(self.figure_path, clf_name))
        plt.clf()
        # TODO should I save the models ?
        # torch.save(clf_model.state_dict(), os.path.join(clf_directory, f'{clf_name}.pt'))
        return clf_model

    def test_preprocess(self, df):
        x = np.array(unpack_batches(self.model.get_embeddings(df['description_base'].tolist())))
        intervention_types = df['intervention_type']

        return x, intervention_types
