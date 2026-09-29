# Notes on ML Installation Guide

Welcome to the **Notes on ML** installation guide! This guide will walk you through setting up the environment, installing necessary dependencies, and configuring essential tools to ensure a smooth development experience.

---

## Prerequisites

Make sure you have the following installed before proceeding:

- **Python**: Version >= 3.9
- **Poetry**: Latest version (for dependency management)

To install Poetry, follow the [official installation guide](https://python-poetry.org/docs/#installation).

---

## 1. Clone and Set Up the Project

First, clone the repository and navigate to the project directory:

```bash
git clone <repository-url>
cd notes-on-ml
```

---

## 2. Install Dependencies with Poetry

Poetry will automatically create a virtual environment and install all dependencies. Run the following command in your project root:

```bash
poetry install
```

This will install all dependencies defined in `pyproject.toml`, including:

- Core dependencies
- Development tools (black, ruff, mypy, etc.)
- Data science packages (pandas, scikit-learn, etc.)
- Visualization tools (matplotlib, seaborn, etc.)
- Testing frameworks (pytest, etc.)

### Optional: Install Specific Groups

You can install specific dependency groups if needed:

```bash
# Install only development dependencies
poetry install --with dev

# Install data science and visualization dependencies
poetry install --with data-science,viz

# Install all groups same as 'poetry install'
poetry install --with dev,test,notebook,data-science,viz
```

---

## 3. Activate the Poetry Environment

To activate the Poetry virtual environment:

```bash
poetry env activate
```

Or run commands directly using:

```bash
poetry run <command>
```

---

## 4. Set Up Development Tools

### Pre-commit Hooks (Optional)

Install pre-commit hooks:

```bash
poetry run pre-commit install
```

This activates pre-commit hooks defined in .pre-commit-config.yaml for your project, ensuring code quality checks run on every commit.

### Jupyter and JupyterLab (Optional)

If you plan to use Jupyter notebooks, install the notebook group:

```bash
poetry install --with notebook
```

To launch JupyterLab:

```bash
poetry run jupyter lab
```

### Set Up Plotly for JupyterLab (Optional)

Install the required JupyterLab extensions for Plotly:

```bash
poetry run jupyter labextension install @jupyter-widgets/jupyterlab-manager@0.36 --no-build
poetry run jupyter labextension install plotlywidget@0.2.1 --no-build
poetry run jupyter labextension install @jupyterlab/plotly-extension@0.16 --no-build
poetry run jupyter lab build
```

---

## 5. Set Up Data Science Tools (Optional)

### Data Version Control (DVC)

If you selected DVC during project creation:

```bash
poetry run dvc init
```

### MLflow

If you selected MLflow during project creation, the tracking server will be available at `http://localhost:5000`:

```bash
poetry run mlflow ui
```

---

## 6. Managing Project Tasks with Invoke

We use **[Invoke](http://www.pyinvoke.org/)** as a task runner for common project management tasks.

### List Available Tasks

```bash
poetry run invoke -l
```

### Get Help on a Specific Task

```bash
poetry run invoke --help <task-name>
```

### Adding Custom Tasks

To add your own tasks, edit the `tasks.py` file in your project root.

---

## 7. Testing

Run the test suite using pytest:

```bash
poetry run pytest
```

For coverage reports:

```bash
poetry run pytest --cov=src
```

---

## 8. Documentation

Build the documentation:

```bash
poetry run mkdocs build
```

Serve the documentation locally:

```bash
poetry run mkdocs serve
```

---

## Final Notes

- Always use `poetry run` to execute commands within the project's virtual environment
- Use `poetry add <package>` to add new dependencies
- Use `poetry update` to update dependencies
- Check `pyproject.toml` for all available dependency groups and their purposes

You're now all set to start developing with **Notes on ML**!
