import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from eval_pipeline.explainers.abstract_explainer import Explainer


def dataset_aspects_to_onehot(dataset):
    """
    Encode the aspects of a dataset in a onehot vector.
    """
    # encode the aspect labels
    dataset = dataset.replace('', -1)
    dataset = dataset.replace('no majority', -1)
    dataset = dataset.replace('Negative', 0)
    dataset = dataset.replace('unknown', 1)
    dataset = dataset.replace('Positive', 2)
    reprs = dataset[['food_aspect_majority', 'service_aspect_majority', 'ambiance_aspect_majority', 'noise_aspect_majority']].astype(int).to_numpy()

    # get a mask for the no majority data
    minus_ones = reprs == -1
    reprs_no_minuses = reprs
    reprs_no_minuses[minus_ones] = 0

    # create onehot encodings for the aspect labels
    N = reprs_no_minuses.shape[0]
    reprs_no_minuses = reprs_no_minuses.flatten()
    reprs_onehot = np.zeros((reprs_no_minuses.size, reprs_no_minuses.max() + 1))
    reprs_onehot[np.arange(reprs_no_minuses.size), reprs_no_minuses] = 1
    reprs_onehot = reprs_onehot.reshape(N, -1, 3)

    # use the mask to set no majority data to the zero vector
    reprs_onehot[minus_ones] = np.zeros(reprs_onehot[minus_ones].shape)

    reprs_onehot = reprs_onehot.reshape(reprs_onehot.shape[0], -1)
    return reprs_onehot


class LIME(Explainer):
    def __init__(self):
        self.model = Pipeline(
            [
                ("lr", LogisticRegression())
            ]
        )

        self.x_train = None
        self.y_train = None
        self.x_dev = None

    def fit(self, dataset, classifier_predictions, classifier, dev_dataset=None):
        # store representations and corresponding model predictions
        self.x_train = dataset_aspects_to_onehot(dataset)
        self.y_train = classifier_predictions

    def _create_distance_matrix(self):
        """
        Calculate the manhattan distance between the dev and train examples.
        """
        N = len(self.x_train)
        M = len(self.x_dev)

        distance = np.zeros((N, M, 2))
        for row in range(distance.shape[0]):
            distance[row, :, 0].fill(row)
        for col in range(distance.shape[1]):
            distance[:, col, 1].fill(col)
        distance = distance.astype(int)

        distance = np.sum(abs(self.x_train[distance.T[0]] - self.x_dev[distance.T[1]]), axis=-1)
        distance = distance.astype(float)

        return distance

    @staticmethod
    def _create_proximity_weight_matrix(distance_matrix):
        """
        Given a distance matrix, calculate the proximity scores based on an exponential kernel.
        """
        sigma = 4
        proximity = np.exp(- distance_matrix ** 2 / sigma ** 2)
        return proximity

    @staticmethod
    def _get_representations_after_interventions(pairs):
        """
        Simulate interventions in the explainable representation space.
        """
        pairs_after_intervention = pairs.copy()
        for aspect in ['food', 'service', 'ambiance', 'noise']:
            pairs_after_intervention[f'{aspect}_aspect_majority_base'] = ((pairs_after_intervention['intervention_type'] == aspect) *
                                                                          pairs_after_intervention['intervention_aspect_counterfactual']) + (
                                                                                     (pairs_after_intervention['intervention_type'] != aspect) *
                                                                                     pairs_after_intervention[f'{aspect}_aspect_majority_base'])

        return dataset_aspects_to_onehot(pairs_after_intervention.rename(columns=lambda col: col.replace('_base', '')))

    def predict_proba(self, pairs):
        # get the aspect encodings for the dev set
        self.x_dev = dataset_aspects_to_onehot(pairs.rename(columns=lambda col: col.replace('_base', '')))

        # apply the correct intervention in the representation space
        counterfactual_dev = self._get_representations_after_interventions(pairs)

        # get proximity matrix
        distance = self._create_distance_matrix()
        proximity = self._create_proximity_weight_matrix(distance)

        # get the model behavior on the train data
        y_train = np.argmax(self.y_train, axis=1)

        # for every dev sample, fit approximations
        n_models = len(self.x_dev)
        print(f'fitting {n_models} LR models...')

        estimates = []
        for i in tqdm(range(n_models)):
            sample_weights = proximity[i]
            lr = self.model.fit(self.x_train, y_train, lr__sample_weight=sample_weights)

            counterfactual = counterfactual_dev[i]
            counterfactual_estimate = lr.predict_proba(counterfactual.reshape(1, -1))
            
            factual = self.x_dev[i]
            factual_estimate = lr.predict_proba(factual.reshape(1, -1))

            estimates.append(np.squeeze(counterfactual_estimate - factual_estimate))
        return estimates
