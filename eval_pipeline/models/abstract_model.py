from abc import ABC, abstractmethod


class Model(ABC):
    def __str__(self):
        return type(self).__name__

    @abstractmethod
    def fit(self, dataset):
        pass

    @abstractmethod
    def predict_proba(self, dataset):
        pass


class DummyModel(Model):
    def fit(self, dataset):
        pass

    def predict_proba(self, dataset):
        dummy_predictions = [[1., 0.]] * len(dataset)
        dummy_report = {'accuracy': 0., 'macro avg': {'f1-score': 0.}}
        return dummy_predictions, dummy_report
