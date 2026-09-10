Tutorial
========

Learn a Domain-Invariant Embedding
----------------------------------

TCA, JDA, and BDA take all samples in a single input array and receive domain
labels through ``covariates``. Use ``target_covariate`` to identify which domain
label is the target domain.

.. code-block:: python

   import numpy as np
   from kalelinear.transformer import TCA

   X = np.array(
       [
           [-2.0, -1.8],
           [-1.8, -2.1],
           [1.9, 1.7],
           [2.1, 2.0],
           [-1.4, -1.2],
           [-1.2, -1.1],
           [1.2, 1.1],
           [1.4, 1.3],
       ]
   )
   domain_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

   transformer = TCA(n_components=2)
   z = transformer.fit_transform(X, covariates=domain_labels, target_covariate=1)

   z_source = z[domain_labels == 0]
   z_target = z[domain_labels == 1]

Align Source and Target Features with CORAL
-------------------------------------------

CORAL is an asymmetric domain alignment method: it whitens the source
covariance and recolors it with the target covariance. As with TCA, pass all
samples in one array with binary domain labels. ``fit_transform`` centers each
domain by its own mean, aligns the source samples, and leaves the target
samples in the same centered space.

.. code-block:: python

   import numpy as np
   from kalelinear.transformer import CORAL

   X = np.array(
       [
           [-2.0, -1.8],
           [-1.8, -2.1],
           [1.9, 1.7],
           [2.1, 2.0],
           [-1.4, -1.2],
           [-1.2, -1.1],
           [1.2, 1.1],
           [1.4, 1.3],
       ]
   )
   domain_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

   transformer = CORAL()
   z = transformer.fit_transform(X, covariates=domain_labels, target_covariate=1)

   z_source = z[domain_labels == 0]
   z_target = z[domain_labels == 1]

Use MIDA with Categorical Covariates
------------------------------------

.. code-block:: python

   import numpy as np
   from kalelinear.transformer import MIDA

   X = np.random.default_rng(0).normal(size=(8, 4))
   y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
   domains = np.array(
       ["source", "source", "source", "source", "target", "target", "target", "target"]
   )

   transformer = MIDA(n_components=2, covariate_encoder="onehot")
   z = transformer.fit_transform(X, y=y, covariates=domains)

Train a Domain Adaptation Classifier
------------------------------------

For ARSVM and ARRLS, pass all source and target samples in ``X``, labels for the
source samples in ``y``, and a covariate vector identifying the target domain.

.. code-block:: python

   import numpy as np
   from kalelinear.estimator import ARSVM

   X = np.array(
       [
           [-2.2, -1.9],
           [-1.9, -2.1],
           [1.8, 2.1],
           [2.0, 1.9],
           [-1.4, -1.2],
           [-1.1, -1.3],
           [1.3, 1.1],
           [1.5, 1.2],
       ]
   )

   source_labels = np.array([0, 0, 1, 1])
   domains = np.array([0, 0, 0, 0, 1, 1, 1, 1])
   X_target = X[domains == 1]

   clf = ARSVM()
   clf.fit(X, source_labels, covariates=domains, target_covariate=1)
   y_pred = clf.predict(X_target)

Train a Manifold-Regularized Classifier
---------------------------------------

LapSVM and LapRLS can use labeled source samples together with unlabeled target
samples. The labels array may contain only the labeled source examples.

.. code-block:: python

   import numpy as np
   from kalelinear.estimator import LapSVM

   X_source = np.array([[-2.0, -1.8], [-1.8, -2.1], [1.9, 1.7], [2.1, 2.0]])
   ys = np.array([0, 0, 1, 1])
   X_target = np.array([[-1.4, -1.2], [-1.2, -1.1], [1.2, 1.1], [1.4, 1.3]])

   X_train = np.vstack((X_source, X_target))

   clf = LapSVM(kernel="linear")
   clf.fit(X_train, ys)
   y_pred = clf.predict(X_target)
