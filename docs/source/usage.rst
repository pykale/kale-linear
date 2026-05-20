Usage
=====

Domain Adaptation Transformers
------------------------------

TCA, JDA, and BDA take all samples in a single input array and receive domain
labels through ``covariates``. Use ``target_covariate`` to identify which domain
label is the target domain.

.. code-block:: python

   import numpy as np
   from kalelinear.transformer import TCA

   x = np.array(
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
   z = transformer.fit_transform(x, covariates=domain_labels, target_covariate=1)

   z_source = z[domain_labels == 0]
   z_target = z[domain_labels == 1]

Domain Adaptation Estimators
----------------------------

ARSVM and ARRLS use all source and target samples in ``x``, labels for the
source samples, and covariates that mark each sample's domain.

.. code-block:: python

   import numpy as np
   from kalelinear.estimator import ARSVM

   x = np.array(
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
   x_target = x[domains == 1]

   clf = ARSVM()
   clf.fit(x, source_labels, covariates=domains, target_covariate=1)
   y_pred = clf.predict(x_target)
