# Project Structure

This document provides a detailed explanation of the project's directory structure and the purpose of each component.

## Directory Structure

```text
.
├── LICENSE
├── README.md              <- The top-level README for developers using this project.
├── CHANGELOG.md           <- A changelog to track project updates and versions.
├── pyproject.toml         <- Project configuration file (replaces setup.py and requirements.txt).
├── .gitignore             <- Specifies intentionally untracked files to ignore.
├── Makefile               <- Automate common tasks like testing, running, or setting up.
├── .env                   <- Environment variables (ignored by git).
├── .pre-commit-config.yaml <- Pre-commit hooks for linting/formatting.
├── app                    <- Main application code (if applicable).
│   └── main.py            <- Entry point for the application.
├── config                 <- Configuration files for the project.
│   ├── dev.yml            <- Development environment configuration.
│   └── prod.yml           <- Production environment configuration.
├── data
│   ├── external           <- Data from third party sources.
│   ├── interim            <- Intermediate data that has been transformed.
│   ├── processed          <- The final, canonical data sets for modeling.
│   └── raw                <- The original, immutable data dump.
├── docs                   <- Project documentation.
│   ├── project_structure.md    <- Project structure tree.
│   ├── install.md         <- Detailed instructions to set up this project.
│   ├── api.md             <- API documentation.
│   ├── user_guide.md      <- User guide for the project.
│   ├── developer_guide.md <- Guide for developers contributing to the project.
│   ├── code_of_conduct.md <- Code of conduct for contributors.
│   └── contributing.md    <- Guidelines for contributing to the project.
├── logs                   <- Log files.
├── models                 <- Trained and serialized models, model predictions, or model summaries.
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description, e.g.
│                             `01-AGE90-initial_data_exploration`.
├── references             <- Data dictionaries, manuals, and all other explanatory materials.
├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures            <- Generated graphics and figures to be used in reporting.
├── scripts                <- Utility scripts for project management, data processing, etc.
│   ├── data_download.sh   <- Script to download raw data.
│   └── setup_env.sh       <- Script to set up the development environment.
├── src
│   └── ml  <- Source code for use in this project.
│       ├── __init__.py    <- Makes ml a Python module.
│       ├── __main__.py    <- Main entry point for the module.
│       ├── credentials.py <- Credentials builder for the project.
│       ├── data           <- Scripts to download or generate data.
│       │   ├── data_loader.py
│       │   └── make_dataset.py
│       ├── features       <- Scripts to turn raw data into features for modeling.
│       │   ├── feature_enineering.py
│       │   └── build_features.py
│       ├── models         <- Scripts to train models and then use trained models to make predictions.
│       │   ├── model_utils.py
│       │   ├── predict_model.py
│       │   └── train_model.py
│       ├── utils          <- Scripts to help with common tasks.
│       │   └── paths.py   <- Helper functions for relative file referencing across project.
│       └── visualization  <- Scripts to create exploratory and results oriented visualizations.
│           └── visualize.py
└── tests                  <- Test files should mirror the structure of `src`.
    ├── __init__.py
    ├── conftest.py        <- Shared pytest fixtures.
    ├── e2e/               <- End-to-end or integration tests.
    └── unit/              <- Unit tests, mirroring src structure.
```

## Key Components

### Top-Level Files

- `LICENSE`: Project license.
- `README.md`: Main project overview and instructions.
- `CHANGELOG.md`: Tracks project updates and versions.
- `pyproject.toml`: Project configuration, dependencies, and tool settings.
- `.gitignore`: Files and folders ignored by Git.
- `Makefile`: Automation for common tasks (testing, setup, etc.).
- `.env`: Environment variables (not tracked by Git).
- `.pre-commit-config.yaml`: Pre-commit hooks for code quality.

### Main Folders

- `app/`: Main application code (if applicable).
  - `main.py`: Application entry point.
- `config/`: Project configuration files.
  - `dev.yml`: Development environment config.
  - `prod.yml`: Production environment config.
- `data/`: Data storage and management.
  - `external/`: Data from third-party sources.
  - `interim/`: Intermediate, transformed data.
  - `processed/`: Final datasets for modeling.
  - `raw/`: Original, immutable data dumps.
- `docs/`: Project documentation.
  - `project_structure.md`: Directory structure and explanations.
  - `install.md`: Installation instructions.
  - `api.md`: API documentation.
  - `user_guide.md`: User documentation.
  - `developer_guide.md`: Developer guidelines.
  - `code_of_conduct.md`: Contributor code of conduct.
  - `contributing.md`: Contribution guidelines.
- `logs/`: Log files.
- `models/`: Trained models, predictions, and summaries.
  - `mlruns/`: MLflow experiment tracking.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
  - Naming convention: `01-AGE90-initial_data_exploration` (number, initials, description).
- `references/`: Data dictionaries, manuals, and explanatory materials.
- `reports/`: Generated analysis (HTML, PDF, LaTeX, etc.).
  - `figures/`: Generated graphics and figures for reporting.
- `scripts/`: Utility scripts for project management and data processing.
  - `data_download.sh`: Script to download raw data.
  - `setup_env.sh`: Script to set up the development environment.
- `src/`: Main source code for the project.
  - `ml/`: Project Python module.
    - `__init__.py`: Module initializer.
    - `__main__.py`: Main entry point for the module.
    - `credentials.py`: Credentials builder.
    - `data/`: Data loading and dataset creation scripts.
      - `data_loader.py`, `make_dataset.py`
    - `features/`: Feature engineering scripts.
      - `feature_engineering.py`, `build_features.py`
    - `models/`: Model training, prediction, and utilities.
      - `model_utils.py`, `predict_model.py`, `train_model.py`
    - `utils/`: Helper functions and utilities.
      - `paths.py`: Relative file referencing helpers.
    - `visualization/`: Visualization scripts.
      - `visualize.py`, `plotting.py`
- `tests/`: Test files mirroring the `src` structure.
  - `__init__.py`: Test module initializer.
  - `conftest.py`: Shared pytest fixtures.
  - `e2e/`: End-to-end/integration tests.
  - `unit/`: Unit tests mirroring `src` structure.

## Best Practices

1. **Data Management**
   - Keep raw data immutable
   - Use DVC for data versioning
   - Document data processing steps

2. **Code Organization**
   - Follow the established directory structure
   - Keep related code together
   - Use appropriate file naming

3. **Documentation**
   - Keep documentation up to date
   - Document all public APIs
   - Include examples in docstrings

4. **Testing**
   - Write tests for all new functionality
   - Maintain good test coverage
   - Use appropriate test fixtures

5. **Version Control**
   - Use meaningful commit messages
   - Follow Git flow branching strategy
   - Keep commits focused and atomic
