Introduction
============

Kale-Linear is a Python library for non-deep, knowledge-aware machine learning
from multiple sources, domains, or views. It provides NumPy-based
implementations of transfer learning, domain adaptation, manifold
regularization, and group-aware linear learning methods with a scikit-learn
style API.

The package is part of the PyKale ecosystem and focuses on classical linear and
kernel methods that are useful when data are structured by domain labels,
covariates, side information, or unlabeled target samples.

Main Features
-------------

* Transformer models for learning feature embeddings: MPCA, TCA, JDA, BDA, and
  MIDA.
* Estimator models for classification and adaptation: LapSVM, LapRLS, ARSVM,
  ARRLS, CoIRSVM, CoIRLS, and GSDA.
* NumPy-compatible inputs and outputs.
* scikit-learn style ``fit``, ``transform``, ``predict``, ``fit_transform``, and
  ``fit_predict`` workflows where applicable.
* Optional covariate encoding for categorical domain or group labels.
