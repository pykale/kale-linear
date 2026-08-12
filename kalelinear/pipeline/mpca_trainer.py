# =============================================================================
# Author: Shuo Zhou, shuo.zhou@sheffield.ac.uk
#         Haiping Lu, h.lu@sheffield.ac.uk or hplu@ieee.org
# =============================================================================

"""Implementation of MPCA->Feature Selection->Linear SVM/LogisticRegression Pipeline

References:
    [1] Swift, A. J., Lu, H., Uthoff, J., Garg, P., Cogliano, M., Taylor, J., ... & Kiely, D. G. (2020). A machine
    learning cardiac magnetic resonance approach to extract disease features and automate pulmonary arterial
    hypertension diagnosis. European Heart Journal-Cardiovascular Imaging.
    [2] Song, X., Meng, L., Shi, Q., & Lu, H. (2015, October). Learning tensor-based features for whole-brain fMRI
    classification. In International Conference on Medical Image Computing and Computer-Assisted Intervention
    (pp. 613-620). Springer, Cham.
    [3] Lu, H., Plataniotis, K. N., & Venetsanopoulos, A. N. (2008). MPCA: Multilinear principal component analysis of
    tensor objects. IEEE Transactions on Neural Networks, 19(1), 18-39.
"""

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.utils.validation import check_is_fitted

from kalelinear.transformer import MPCA

param_c_grids = list(np.logspace(-4, 2, 7))
classifiers = {
    "svc": [SVC, {"kernel": ["linear"], "C": param_c_grids, "max_iter": [50000]}],
    "linear_svc": [LinearSVC, {"C": param_c_grids}],
    "lr": [LogisticRegression, {"C": param_c_grids}],
}

# k-fold cross-validation used for grid search, i.e. searching for optimal value of C
default_search_params = {"cv": 5}
default_mpca_params = {"explained_variance_ratio": 0.97, "vectorize": True}


class MPCATrainer(BaseEstimator, ClassifierMixin):
    """Trainer of pipeline: MPCA->Feature selection->Classifier

    Args:
        classifier (str, optional): Available classifier options: {"svc", "linear_svc", "lr"}, where "svc" trains a
            support vector classifier, supports both linear and non-linear kernels, optimizes with library "libsvm";
            "linear_svc" trains a support vector classifier with linear kernel only, and optimizes with library
            "liblinear", which suppose to be faster and better in handling large number of samples; and "lr" trains
            a classifier with logistic regression. Defaults to "svc".
        classifier_params (dict, optional): Parameters of classifier. Defaults to 'auto'.
        classifier_param_grid (dict, optional): Grids for searching the optimal hyper-parameters. Works only when
            classifier_params == "auto". Defaults to None by searching from the following hyper-parameter values:
            1. svc, {"kernel": ["linear"], "C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100], "max_iter": [50000]},
            2. linear_svc, {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]},
            3. lr, {"C": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
        mpca_params (dict, optional): Parameters of MPCA, e.g., {"explained_variance_ratio": 0.8}. Defaults to None,
            i.e., using the default parameters
            (https://kalelinear.readthedocs.io/en/latest/api_transformers.html#kalelinear.transformer.MPCA).
        n_features (int, optional): Number of features for feature selection. Defaults to None, i.e., all features
            after dimension reduction will be used.
        search_params (dict, optional): Parameters of grid search, for more detail please see
            https://scikit-learn.org/stable/modules/grid_search.html#grid-search. Defaults to None, i.e., using the
            default params: {"cv": 5}.
    """

    def __init__(
        self,
        classifier="svc",
        classifier_params="auto",
        classifier_param_grid=None,
        mpca_params=None,
        n_features=None,
        search_params=None,
    ):
        if classifier not in ["svc", "linear_svc", "lr"]:
            error_msg = "Valid classifier should be 'svc', 'linear_svc', or 'lr', but given %s" % classifier
            logging.error(error_msg)
            raise ValueError(error_msg)

        self.classifier = classifier
        # init mpca object
        if mpca_params is None:
            self.mpca_params = dict(default_mpca_params)
        else:
            self.mpca_params = dict(mpca_params)
        self.mpca = MPCA(**self.mpca_params)
        # init feature selection parameters
        self.n_features = n_features
        # init classifier object
        if search_params is None:
            self.search_params = dict(default_search_params)
        else:
            self.search_params = dict(search_params)
        self.classifier_param_grid = classifier_param_grid

        self.auto_classifier_param = False
        if classifier_params == "auto":
            self.auto_classifier_param = True
            if self.classifier_param_grid is None:
                self.classifier_param_grid = {
                    param_name: list(values) for param_name, values in classifiers[classifier][1].items()
                }
            self.grid_search = GridSearchCV(
                classifiers[classifier][0](), param_grid=self.classifier_param_grid, **self.search_params
            )
            self.clf = None
        elif isinstance(classifier_params, dict):
            self.clf = classifiers[classifier][0](**classifier_params)
        else:
            error_msg = "Invalid classifier parameter type"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if isinstance(classifier_params, dict):
            self.classifier_params = dict(classifier_params)
            self.clf = classifiers[classifier][0](**classifier_params)
        elif classifier_params == "auto":
            self.auto_classifier_param = True
            self.classifier_params = "auto"
        else:
            error_msg = "Invalid classifier parameter type"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def fit(self, X, y):
        """Fit a pipeline with the given data X and labels y

        Args:
            X (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)
            y (array-like): data labels, shape (n_samples, )

        Returns:
            self
        """
        # fit mpca
        self.mpca.fit(X)
        self.mpca.set_params(**{"vectorize": True})
        X_transformed = self.mpca.transform(X)

        # feature selection
        if self.n_features is None:
            self.n_features_ = X_transformed.shape[1]
            self.feature_order_ = self.mpca.idx_order_
        else:
            f_score, p_val = f_classif(X_transformed, y)
            self.feature_order_ = (-1 * f_score).argsort()
            self.n_features_ = self.n_features
        X_transformed = X_transformed[:, self.feature_order_][:, : self.n_features_]

        # fit classifier
        if self.auto_classifier_param:
            param_grid = {name: list(values) for name, values in self.classifier_param_grid.items()}
            extra_c = 1 / X.shape[0]
            if extra_c not in param_grid["C"]:
                param_grid["C"].append(extra_c)
            self.grid_search = GridSearchCV(
                classifiers[self.classifier][0](), param_grid=param_grid, **self.search_params
            )
            self.grid_search.fit(X_transformed, y)
            self.clf = self.grid_search.best_estimator_
        if self.classifier == "svc":
            self.clf.set_params(**{"probability": True})

        self.clf.fit(X_transformed, y)
        return self

    def predict(self, X):
        """Predict the labels for the given data X

        Args:
            X (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: Predicted labels, shape (n_samples, )
        """
        features = self._extract_feature(X)
        return self.clf.predict(features)

    def decision_function(self, X):
        """Decision scores of each class for the given data X

        Args:
            X (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: decision scores, shape (n_samples,) for binary case, else (n_samples, n_classes)
        """
        features = self._extract_feature(X)
        return self.clf.decision_function(features)

    def predict_proba(self, X):
        """Probability of each class for the given data X. Not supported by "linear_svc".

        Args:
            X (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: probabilities, shape (n_samples, n_classes)
        """
        if self.classifier == "linear_svc":
            error_msg = "Linear SVC does not support computing probability."
            logging.error(error_msg)
            raise ValueError(error_msg)
        features = self._extract_feature(X)
        return self.clf.predict_proba(features)

    def _extract_feature(self, X):
        """Extracting features for the given data X with MPCA->Feature selection

        Args:
            X (array-like tensor): input data, shape (n_samples, I_1, I_2, ..., I_N)

        Returns:
            array-like: n_new, shape (n_samples, n_features)
        """
        check_is_fitted(self)
        X_transformed = self.mpca.transform(X)

        return X_transformed[:, self.feature_order_][:, : self.n_features_]
