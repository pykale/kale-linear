# Tutorials

Worked examples for the `kalelinear` transformers and estimators. For
installation instructions and an API overview, see the
[README](README.md).

## Learn a Domain-Invariant Embedding

### Use TCA for Two-Domain Adaptation

```python
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
```

TCA, JDA, and BDA take domain labels through `covariates`. They do not accept
separate source and target arrays; stack samples into one array and use
`target_covariate` to identify the target domain.

### Align Source and Target Features with CORAL

CORAL is an asymmetric domain alignment method: it whitens the source
covariance and recolors it with the target covariance. As with TCA, pass all
samples in one array with binary domain labels. `fit_transform` centers each
domain by its own mean, aligns the source samples, and leaves the target
samples in the same centered space.

```python
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
```

### Use MIDA with Categorical Domain Covariates

```python
import numpy as np
from kalelinear.transformer import MIDA

X = np.random.default_rng(0).normal(size=(8, 4))
y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
domains = np.array(["source1", "source1", "source2", "source2", "target", "target", "target", "target"])

transformer = MIDA(n_components=2, covariate_encoder="onehot")
z = transformer.fit_transform(X, y=y, covariates=domains)
```

## Train a Domain Adaptation Classifier

For ARSVM and ARRLS, pass all source and target samples in `X`, labels for the
source samples in `y`, and a covariate vector identifying the target domain.

```python
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
```

## Train a Manifold-Regularized Classifier

LapSVM and LapRLS can use labeled source samples together with unlabeled target
samples. The labels array may contain only the labeled source examples.

```python
import numpy as np
from kalelinear.estimator import LapSVM

X_source = np.array([[-2.0, -1.8], [-1.8, -2.1], [1.9, 1.7], [2.1, 2.0]])
ys = np.array([0, 0, 1, 1])
X_target = np.array([[-1.4, -1.2], [-1.2, -1.1], [1.2, 1.1], [1.4, 1.3]])

X_train = np.vstack((X_source, X_target))

clf = LapSVM(kernel="linear")
clf.fit(X_train, ys)
y_pred = clf.predict(X_target)
```

## Learn Common and Individual Features Across Groups

CIFE and AJIVE decompose multiblock data into a common feature subspace shared
by all blocks plus block-specific individual subspaces. Each block below is
generated from a shared two-dimensional common basis plus a block-specific
three-dimensional individual basis, so every 50x10 block is rank-deficient and
the common structure is identifiable. Blocks can be passed either as a stacked
matrix with a `groups` array of block ids or as a list of block matrices
sharing the same feature space.

```python
import numpy as np
from kalelinear.transformer import CIFE, AJIVE

rng = np.random.default_rng(0)
n_common, n_individual = 2, 3

# Shared common basis plus block-specific individual bases, so each block only
# spans 2 + 3 = 5 of the 10 feature dimensions.
common_basis, _ = np.linalg.qr(rng.normal(size=(10, n_common)))
blocks = []
for _ in range(3):
    individual_basis, _ = np.linalg.qr(rng.normal(size=(10, n_individual)))
    individual_basis -= common_basis @ (common_basis.T @ individual_basis)
    individual_basis, _ = np.linalg.qr(individual_basis)
    block = rng.normal(size=(50, n_common)) @ common_basis.T
    block += rng.normal(size=(50, n_individual)) @ individual_basis.T
    blocks.append(block)

X = np.vstack(blocks)
groups = np.repeat([0, 1, 2], 50)

cife = CIFE(n_individual_components=n_individual, random_state=0)
z_common = cife.fit_transform(X, groups=groups)
z_individual = cife.transform_individual(X, groups=groups)

ajive = AJIVE(n_resamples=50, random_state=0)
z_common_ajive = ajive.fit_transform(X, groups=groups)
```

`transform` returns the common feature scores of the samples (two common
components are recovered here), and `transform_individual` returns a list with
the block-specific individual scores (three per block).
