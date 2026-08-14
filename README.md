<p align="center">
<img src="https://github.com/pykale/linear/raw/main/docs/images/kalelinear.jpg" width="60%" alt="kalelinear logo" />
</p>

[![tests](https://github.com/pykale/linear/workflows/test/badge.svg)](https://github.com/pykale/linear/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/pykale/linear/branch/main/graph/badge.svg)](https://codecov.io/gh/pykale/linear)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/pykale/linear/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org)
[![PyPI version](https://img.shields.io/pypi/v/kalelinear?color=blue)](https://pypi.org/project/kalelinear/)
[![PyPI downloads](https://pepy.tech/badge/kalelinear)](https://pepy.tech/project/kalelinear)

`kalelinear` is a Python library for learning harmonized or individualized models from multi-source/multi-view data in linear or reproducing kernel Hilbert spaces (RKHS). It provides NumPy-based methods for leveraging related data distributions and structural assumptions, including transfer learning, domain adaptation, manifold regularization, and group-aware learning, through a [`scikit-learn`](https://github.com/scikit-learn/scikit-learn) style API.

The package is part of the [PyKale](https://github.com/pykale/pykale) ecosystem and focuses on linear and kernel methods for data characterized by covariates (e.g., domain labels, group labels, side information), unlabeled target samples, or tensor structures.

## What's included

- Transformer models for learning feature embeddings:
  - Multilinear Principal Component Analysis (MPCA) [[1](#references)]
  - Transfer Component Analysis (TCA) [[2](#references)]
  - Joint Distribution Adaptation (JDA) [[3](#references)]
  - Balanced Distribution Adaptation (BDA) [[4](#references)]
  - Maximum Independence Domain Adaptation (MIDA) [[5](#references)]
- Estimator models for classification:
  - Manifold Regularization Learning Framework (LapSVM, LapRLS) [[6](#references)]
  - Adaptation Regularization Learning Framework (ARSVM, ARRLS) [[7](#references)]
  - Covariate Independence Regularized Learning Framework (CoIRSVM, CoIRLS) [[8](#references)][[9](#references)]
  - Group-specific Discriminant Analysis (GSDA) [[9](#references)][[10](#references)]
- NumPy-compatible inputs and outputs.
- scikit-learn style `fit`, `transform`, `predict`, `fit_transform`, and `fit_predict` workflows where applicable.
- Optional covariate encoding for categorical domain or group labels.

`kalelinear` requires Python 3.10 or later. Core dependencies include:

- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [tensorly](http://tensorly.org/)
- [cvxopt](http://cvxopt.org/)
- [osqp](https://osqp.org/)

## Getting started

### Installation

Install the released package from PyPI:

```bash
pip install kalelinear
```

Install from a local checkout for development:

```bash
pip install -e ".[dev]"
```

### Development

From the root of the repository, run the following commands in your terminal:

1. Install pre-commit hooks (only required once):

   ```bash
   pre-commit install
   ```

2. Run pre-commit checks for code style and formatting on all files:

   ```bash
   pre-commit run --all-files
   ```

3. Run test cases to verify functionality:

   ```bash
   pytest
   ```

4. Build the documentation:

   ```bash
   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html
   ```

### Public API

```python
from kalelinear.transformer import BDA, JDA, MIDA, MPCA, TCA
from kalelinear.estimator import ARRLS, ARSVM, CoIRLS, CoIRSVM, GSDA, LapRLS, LapSVM
```

Worked examples for the main transformers and estimators are collected in
[Tutorials](TUTORIALS.md):

- Learn a domain-invariant embedding with TCA
- Use MIDA with categorical covariates
- Train a domain adaptation classifier (ARSVM, ARRLS)
- Train a manifold-regularized classifier (LapSVM, LapRLS)

# References

[1] Lu, H., Plataniotis, K.N. and Venetsanopoulos, A.N., 2008. [MPCA: Multilinear principal component analysis of tensor objects](https://ieeexplore.ieee.org/abstract/document/4359192/). _IEEE Transactions on Neural Networks_, 19(1), pp.18-39.

[2] Pan, S.J., Tsang, I.W., Kwok, J.T. and Yang, Q., 2011. [Domain adaptation via transfer component analysis](https://dl.acm.org/doi/abs/10.1109/TNN.2010.2091281). _IEEE Transactions on Neural Networks_, 22(2), p.199-210.

[3] Long, M., Wang, J., Ding, G., Sun, J. and Yu, P.S., 2013. [Transfer feature learning with joint distribution adaptation.](https://openaccess.thecvf.com/content_iccv_2013/papers/Long_Transfer_Feature_Learning_2013_ICCV_paper.pdf) In _Proceedings of the IEEE International Conference on Computer Vision_ (pp. 2200-2207).

[4] Wang, J., Chen, Y., Hao, S., Feng, W. and Shen, Z., 2017, November. [Balanced distribution adaptation for transfer learning](https://ieeexplore.ieee.org/document/8215613). In _2017 IEEE International Conference on Data Mining (ICDM)_ (pp. 1129-1134). IEEE.

[5] Yan, K., Kou, L. and Zhang, D., 2017. [Learning domain-invariant subspace using domain features and independence maximization](https://ieeexplore.ieee.org/document/7815350). _IEEE Transactions on Cybernetics_, 48(1), pp.288-299.

[6] Belkin, M., Niyogi, P. and Sindhwani, V., 2006. Manifold regularization: [A geometric framework for learning from labeled and unlabeled examples](https://www.jmlr.org/papers/v7/belkin06a.html). _Journal of Machine Learning Research_, 7(11).

[7] Long, M., Wang, J., Ding, G., Pan, S.J. and Yu, P.S., 2013. [Adaptation regularization: A general framework for transfer learning](https://ieeexplore.ieee.org/abstract/document/6550016/). _IEEE Transactions on Knowledge and Data Engineering_, 26(5), pp.1076-1089.

[8] Zhou, S., Li, W., Cox, C. and Lu, H., 2020, April. [Side information dependence as a regularizer for analyzing human brain conditions across cognitive experiments](https://ojs.aaai.org/index.php/AAAI/article/view/6179). In _Proceedings of the AAAI Conference on Artificial Intelligence_ (Vol. 34, No. 04, pp. 6957-6964).

[9] Zhou, S., 2022. [Interpretable Domain-Aware Learning for Neuroimage Classification](https://etheses.whiterose.ac.uk/id/eprint/31044/) (Doctoral dissertation, University of Sheffield).

[10] Zhou, S., Luo, J., Jiang, Y., Wang, H., Lu, H. and Gong, G., 2025. [Group-specific discriminant analysis enhances detection of sex differences in brain functional network lateralization](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf082/8244707). _GigaScience_, 14, p.giaf082.

## Other open domain adaptation repositories

- [POT: Python Optimal Transport](https://github.com/rflamary/POT)
- [Everything about Transfer Learning](https://github.com/jindongwang/transferlearning)
- [ADA: Another Domain Adaptation library](https://github.com/criteo-research/pytorch-ada)
- [Domain Adaptation and Transfer Learning Repositories](https://github.com/domainadaptation)
- [Library of transfer learners and domain-adaptive classifiers](https://github.com/wmkouw/libTLDA)
- [domain-adaptation-toolbox](https://github.com/viggin/domain-adaptation-toolbox)
- [Domain-Adaptations](https://github.com/wihoho/Domain-Adaptations)

## License

`kalelinear` is released under the MIT License. See [LICENSE](LICENSE) for details.
