Installation
============

Install the released package from PyPI:

.. code-block:: bash

   pip install kalelinear

Install a local checkout for development:

.. code-block:: bash

   pip install -e .[dev]

Kale-Linear requires Python 3.10 or later. Core dependencies include NumPy,
SciPy, scikit-learn, pandas, tensorly, cvxopt, and osqp.

To build the documentation locally, install the documentation requirements and
run Sphinx from the repository root:

.. code-block:: bash

   pip install -r docs/requirements.txt
   sphinx-build -b html docs/source docs/build/html
