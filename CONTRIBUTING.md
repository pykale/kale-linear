# Contributing to kalelinear

[Light involvements (viewers/users)](#light-involvements-viewersusers) |
[*Medium involvements (contributors)*](#medium-involvements-contributors) |
[**Heavy involvements (maintainers)**](#heavy-involvements-maintainers)

[Ask questions](#ask-questions) |
[Report bugs](#report-bugs) |
[Suggest improvements](#suggest-improvements) |
[*Branch, fork & pull*](#branch-fork-and-pull) |
[*Coding style*](#coding-style) |
[*Test*](#testing) |
[Review & merge](#review-and-merge-pull-requests) |
[Release & management](#release-and-management)

Thank you for your interest! You can contribute to `kalelinear` in a wide range of ways listed above, from light to heavy involvements. You can also reach us via <a href="mailto:pykale-group&#64;sheffield.ac.uk">email</a> if needed. Participation in this open source project is subject to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Light involvements (viewers/users)

See the [ReadMe](README.md) for installation instructions. Your contribution can start as light as asking questions.

### Ask questions

Ask any questions about `kalelinear` on the [PyKale GitHub Discussions tab](https://github.com/pykale/pykale/discussions) and we will discuss and answer them there. Questions help us identify *blind spots* in our development and can greatly improve code quality.

### Report bugs

Search current issues to see whether they are already reported. If not, report bugs by [creating issues](https://github.com/pykale/linear/issues) using the provided template. Even better, if you know how to fix them, make suggestions and/or propose changes with [pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).

### Suggest improvements

Suggest possible improvements such as new features or code refactoring by [creating issues](https://github.com/pykale/linear/issues) using the respective templates. Even better, you are welcome to propose such changes with [pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/proposing-changes-to-your-work-with-pull-requests).

## Medium involvements (contributors)

We use **US English spelling** and recommend spell check via [Grazie](https://github.com/JetBrains/intellij-community/tree/master/plugins/grazie) in PyCharm and [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) in VS Code with the US English setting.

### Branch, fork and pull

A maintainer with *write* access can [create a branch](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository) directly in `pykale/linear` to make changes under the [shared repository model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models), following the steps below while skipping the fork step.

Anyone can use the [*fork and pull* model](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models) to contribute code to `kalelinear`:

- [**Fork**](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) `pykale/linear` (also see the [guide on forking projects](https://guides.github.com/activities/forking/)).
  - Keep the fork `main` branch synced with `pykale/linear:main` by [syncing a fork](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork).
  - Install `pre-commit` to enforce style via `pip install pre-commit` and `pre-commit install` at the root.
- [Create a branch](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-and-deleting-branches-within-your-repository) based on the *latest main* in your fork with a *descriptive* name on what you plan to do, e.g. to fix an issue, starting with the issue ticket number.
  - Make changes to this branch using detailed commit messages and following the [coding style](#coding-style) below. In particular, do [**frequent commits**](https://docs.github.com/en/actions/guides/about-continuous-integration#about-continuous-integration) and keep pull requests **small-scale** to make them more focused and easier to review.
  - [Sync your branch](https://docs.github.com/en/desktop/contributing-and-collaborating-using-github-desktop/syncing-your-branch) with `main` frequently so that potential problems can be identified earlier.
  - Document the update in [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). Update `docs` following the steps in [Documentation](#documentation) below.
  - Build tests and do tests following [Testing](#testing) below.
- Create a [draft pull request](https://github.blog/2019-02-14-introducing-draft-pull-requests/) or [pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request) from the task branch above to the `main` branch `pykale/linear:main`, explaining the changes and choosing reviewers, using the [template](#pull-request-template).
  - A draft pull request helps start a conversation with collaborators in a draft state. It will not be reviewed or merged until you change the status to "Ready for review" near the bottom of your pull request.
  - View the [continuous integration (CI) status checks](https://github.com/pykale/linear/actions). When the check messages say files are changed, they mean changes on their simulated environment, *NOT* on the branch. The problems are not fixed and you need to fix them as well as other reported errors locally.
  - You need to [address merge conflicts](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/addressing-merge-conflicts) if they arise. Resolve the conflicts locally.
  - After passing all CI checks and resolving the conflicts, [request a review](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review). If you know who is appropriate or like the [suggested reviewers](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/requesting-a-pull-request-review), request/assign that person. Otherwise, we will assign one shortly.
  - A reviewer will follow the [review and merge guidelines](#review-and-merge-pull-requests). The reviewer may discuss with you and request explanations/changes before merging.
  - Merging to the `main` branch **requires** *ALL checks to pass* AND *at least one approving review*.

#### Before pull requests: pre-commit hooks

We set up several [`pre-commit`](https://pre-commit.com/) hooks to ensure code quality, including:

- Linting tools: [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort).
- Static type analysis: [mypy](https://github.com/python/mypy).
- Other hooks as specified in [`.pre-commit-config.yaml`](.pre-commit-config.yaml), such as restricting the largest file size to 300KB.

You need to install `pre-commit` and the hooks from the root directory via:

```bash
pip install pre-commit
pre-commit install
```

This installs the `pre-commit` hooks at `kalelinear\.git\hooks`, to be **triggered by each new commit** to automatically run them *over the files you commit*. Several **important** points to note:

- Pre-commit hooks are configured in [`.pre-commit-config.yaml`](.pre-commit-config.yaml). Only administrators should modify it.
- These hooks, e.g. [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/), will **automatically fix** some problems by **changing the files**, so please check the changes after you trigger `commit`.
- If your commits cannot pass the above checks, read the error message to see what has been automatically fixed and what needs your manual fix, e.g. flake8 errors. You can rerun pre-commit (e.g. re-commit to trigger it) or just run flake8 to see the updated errors.

#### Manual checks and fixes (be *CAREFUL*)

Required libraries will be automatically installed but if you wish, you may install them manually and run them **from the root directory** (so that the `kalelinear` configurations are used). For example:

```bash
pip install black # The first time
black ./kalelinear/new_module.py # "black ." does it for all files
pip install isort # The first time
isort ./kalelinear/new_module.py # "isort ." does it for all files
pip install flake8 # The first time
flake8 ./kalelinear/new_module.py # "flake8 ." does it for all files
```

Run [black](https://black.readthedocs.io/en/stable/index.html) and [isort](https://pycqa.github.io/isort/) to fix the found problems automatically by modifying the files; remaining [flake8](https://flake8.pycqa.org/en/latest/) or other errors need to be manually fixed.

**Important**: Run these commands from the root directory so that the `kalelinear` configuration files ([`setup.cfg`](setup.cfg), [`pyproject.toml`](pyproject.toml), and [`.pre-commit-config.yaml`](.pre-commit-config.yaml)) are used for these tools. Otherwise, the default configurations will be used, which differ from the `kalelinear` configurations.

**IDE integration**: flake8 linting can be set up for both [VSCode](https://code.visualstudio.com/docs/python/linting) and [PyCharm](https://tirinox.ru/flake8-pycharm/), but you must use [`setup.cfg`](setup.cfg) to configure it. In this way, you can fix linting errors on the go.

#### Automated GitHub workflows (continuous integration)

For continuous integration (CI) and continuous deployment (CD), we use several [GitHub workflows (actions)](https://github.com/pykale/linear/actions) triggered upon a push or pull request as specified in [`pykale/linear/.github/workflows/`](.github/workflows):

- Tests: unit and regression tests with coverage reporting.
- Linting: pre-commit checks including flake8, black, and isort.
- Release: build and publish the package to PyPI upon a GitHub release.
- Changelog: automatically generate changelog entries on release pull requests.
- Security: CodeQL analysis.

#### Pull request template

We have a pull request template. Please use it for all pull requests and mark the status of your pull requests:

- **Ready**: ready for review and merge (if no problems found). Reviewers will be assigned.
- **Work in progress**: for core team awareness of this development (e.g. to avoid duplicated efforts) and possible feedback (e.g. to find problems early, such as linting/CI issues). Not ready to merge yet. Change it to **Ready** when ready to merge.
- **Hold**: not for attention yet.

### Coding style

We aim to design the core `kalelinear` modules to be highly **reusable**, generic, and customizable, and follow these guidelines:

- Follow the [continuous integration practice](https://docs.github.com/en/actions/guides/about-continuous-integration#about-continuous-integration) to make small changes and commit frequently with clear descriptions for others to understand what you have done. This detects errors sooner, reduces debugging, makes merging easier, and eventually saves overall time.
- Use highly *readable* names for variables, functions, and classes. Using *verbs* is preferred when feasible for compactness. Use spell check with the **US** English setting, e.g. [Grazie](https://github.com/JetBrains/intellij-community/tree/master/plugins/grazie) in PyCharm and [Code Spell Checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) in VS Code.
- Use [`logging`](https://docs.python.org/3/howto/logging.html#logging-basic-tutorial) instead of `print` to log messages. Users can choose the level via, e.g., `logging.getLogger().setLevel(logging.INFO)`.
- Include detailed docstrings in code for generating documentation, following the [Google Style Python Docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
- Highly reusable modules should go into `kalelinear`. Highly data/example-specific code goes into `examples`.
- Provide a [scikit-learn](https://scikit-learn.org/) style API with `fit`, `transform`, `predict`, `fit_transform`, and `fit_predict` workflows where applicable, using NumPy arrays for inputs and outputs.
- If high-quality existing code from other sources is used, add credit and license information at the top of the file.
- Use pre-commit hooks to enforce consistent styles via [flake8](https://gitlab.com/pycqa/flake8), [black](https://github.com/psf/black), and [isort](https://github.com/timothycrosley/isort), with the common `kalelinear` configuration files.

#### Recommended development software

- Python IDE: [Visual Studio Code](https://code.visualstudio.com/download), [PyCharm](https://www.jetbrains.com/pycharm/download/)
- GitHub: [GitHub Desktop (for Windows/Mac)](https://desktop.github.com/), [GitKraken (for Linux)](https://www.gitkraken.com/), [GitHub guides](https://guides.github.com/), [GitHub documentations](https://docs.github.com/en)

### Documentation

Documentation is built with [Sphinx](https://www.sphinx-doc.org/) from `docs/source` using the `sphinx_rtd_theme`:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs/source docs/build/html
```

Verify the locally built documentation under `docs/build/html`. Update the API documentation when you add or change public modules, classes, or functions.

### Testing

All new code should be covered by [unit tests](https://carpentries-incubator.github.io/python-testing/04-units/index.html), and [regression tests](https://carpentries-incubator.github.io/python-testing/07-integration/index.html) where appropriate. We will extend test coverage to existing code.

Please use the [PyKale discussions on testing](https://github.com/pykale/pykale/discussions/categories/testing) to talk about tests and ask for help.

Refer to the [official pytest documentation](https://docs.pytest.org/en/stable/) if needed.

#### Test runner

`kalelinear` uses the `pytest` test runner. This offers a balance of functionality, ease of use, and wide community support.

#### Unit tests

A **unit test** checks that a small "unit" of software (e.g. a function) performs correctly. It might, for example, check that the function `add` returns the number `2` when a list `[1, 1]` is the input.

Within the `tests/` folder is a folder structure that mimics that of the `kalelinear` Python module. Unit tests for code in a given file in `kalelinear/` should be placed in their equivalent file in `tests/`, e.g. unit tests for a function in `kalelinear/estimator/_gsda.py` should be located in `tests/estimator/test_gsda.py`.

Philosophically, the author of a "unit" of code knows exactly what it should do and can write the test criteria accordingly.

#### Regression tests

A **regression test** checks that software produces the same results after a change is made. In `kalelinear`, we expect regression tests to achieve this by testing several different parts of the software at once (in effect, an [integration test](https://carpentries-incubator.github.io/python-testing/07-integration/index.html)). A single regression test might test *loading some input files*, *setting up a model*, and *generating a prediction or plot* based on the model. This could be achieved by running the software with previously stored baseline inputs and checking the output is the same as previously stored baseline outputs.

Regression tests should be placed in `tests/regression`. Further subfolders can be added as required.

#### Test data

Data needed for testing should be placed in `tests/test_data`. This should be limited to small text files, e.g. `.csv`, `.json`, `.yml`. Binary data should be stored outside the repository and referenced, e.g. using a DOI. Discuss more complex test data requirements for your **pull request** in the motivating **issue**.

#### Common parameters

Consider adding parameters (or objects etc.) that may be useful to multiple tests as fixtures in a [`conftest.py`](https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixtures-across-multiple-files) file, either in `tests/` or the appropriate sub-module.

#### Testing DataFrames and arrays

Comparisons/assertions involving `pandas` `DataFrames` (or other `pandas` objects) should be made using `pandas` utility functions: [`pandas.testing.assert_frame_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_frame_equal.html), [`pandas.testing.assert_series_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_series_equal.html), [`pandas.testing.assert_index_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_index_equal.html), and [`pandas.testing.assert_extension_array_equal`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.testing.assert_extension_array_equal.html).

Comparisons/assertions involving `numpy` `arrays` (or other `numpy` objects) should be made using [`numpy` testing routines](https://numpy.org/doc/stable/reference/routines.testing.html). `numpy` floating point "problem" response will be [as default](https://numpy.org/doc/stable/reference/generated/numpy.seterr.html#numpy.seterr).

#### Random numbers

Random numbers in `kalelinear` are generated using base Python and `numpy`. Prior to making an assertion where objects that make use of random numbers are compared, set a seed via `numpy.random.seed()` (or a fixture) so that tests are reproducible.

#### Logging and handling of warnings

`pytest` [captures log messages of level WARNING or above](https://docs.pytest.org/en/stable/logging.html) and outputs them to the terminal.

#### Side effects

Be aware that the code for which you are adding a test may have [side effects](https://en.wikipedia.org/wiki/Side_effect_(computer_science)) (e.g. a function changing something in a file or database, as well as returning a variable). Minimize side effects and ensure, where present, they are covered by tests.

## Heavy involvements (maintainers)

### Review and merge pull requests

A maintainer assigned to review a pull request should follow GitHub guidelines on how to [review changes in pull requests](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/reviewing-changes-in-pull-requests) and [incorporate changes from a pull request](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/incorporating-changes-from-a-pull-request). Merging can be automated, in which case an approving review will trigger the merging. You should NOT approve changes if they are not ready to merge.

If you think you are not the right person to review, let an administrator know for a reassignment. If multiple reviewers are assigned, anyone can approve and merge unless more approvals are explicitly required.

For simple problems, such as typos or hyperlinks, reviewers can fix them directly and push the changes rather than comment and wait for the author to fix them. This speeds up development.

### Release and management

Releases are created manually in GitHub, with automatic upload to PyPI by the [release workflow](.github/workflows/release.yml) (prereleases are published to Test PyPI).

#### Versions

We follow the [Semantic Versioning](https://semver.org/) guidelines. Given a version number `MAJOR.MINOR.PATCH`, increment the:

- MAJOR version when you make incompatible API changes,
- MINOR version when you add functionality in a backwards-compatible manner, and
- PATCH version when you make backwards-compatible bug fixes.

Additional labels for pre-release and build metadata are available as extensions to the `MAJOR.MINOR.PATCH` format.

#### Release checklist

- Bump the version in `setup.py` and `kalelinear/__init__.py`.
- Update [`.github/CHANGELOG.md`](.github/CHANGELOG.md) with a summary of changes since the last release.
- Create a GitHub release for the new version (e.g. `0.1.0b1`), marking prereleases appropriately.
- The [release workflow](.github/workflows/release.yml) builds the wheel and source distribution and publishes them to Test PyPI (prerelease) or PyPI (final release).

## References

The following libraries are good resources to learn from:

- [scikit-learn](https://github.com/scikit-learn/scikit-learn): machine learning in Python with a consistent estimator API
- [NumPy](https://github.com/numpy/numpy): fundamental package for scientific computing in Python
- [SciPy](https://github.com/scipy/scipy): fundamental algorithms for scientific computing in Python
- [TensorLy](https://github.com/tensorly/tensorly): tensor learning in Python
