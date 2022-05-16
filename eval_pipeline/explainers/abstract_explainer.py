from abc import ABC, abstractmethod


class Explainer(ABC):
    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        pass

    @abstractmethod
    def predict_proba(self, pairs):
        pass


class DummyExplainer(Explainer):
    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        pass

    def predict_proba(self, pairs):
        return [[1., 0.]] * len(pairs)
