# Version 0.1.0b1

#### New Features

* MPCA updated to a new version more compatible with NumPy/scikit-learn workflows, with a memory-efficient implementation and a new `MPCATrainer` pipeline.
* Added `embed` and `predict` modules to the Kale API.
* Added a group-specific discriminant analysis (GSDA) example for brain lateralization analysis.
* Added Sphinx documentation and ReadTheDocs configuration.
* Added detailed contributing guidelines, a code of conduct, and issue and pull request templates.

#### Code Improvements

* Refactored estimators (ARL, CoIR, manifold learning, GSDA) with a unified base class, consistent variable naming, and improved docstrings.
* Removed the PyTorch dependency; core methods now run on NumPy only.
* Updated `setup.py` for default, extras (`full`), and development (`dev`) installation options.

#### Bug Fixes

* Fixed optimization issues in the shared estimator and transformer base classes: the QP Hessian is projected onto the positive-semidefinite cone and generalized eigenproblems are regularized, so solvers such as `osqp` and `scipy` no longer fail on semidefinite inputs.

#### Documentation

* Updated the README, tutorials, and API documentation for the `kalelinear` package.

#### Tests

* Set up the test file structure and strategy.
* Added tests for the public API, MPCA and `MPCATrainer`, estimators, transformers, and utilities.
* Added Codecov coverage reporting to the test workflow.

# Version  0.1.0a1

#### New Features

* Initial release of `kalelinear`: knowledge-aware linear and kernel methods for multi-source/multi-view learning, including transfer learning, domain adaptation, manifold regularization, and group-aware estimators and transformers.
