# User Guide

This guide provides detailed instructions for using the **Notes on ML** project.

## Getting Started

### Installation

1. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

2. Clone the repository:
   ```bash
   git clone <repository-url>
   cd notes-on-ml
   ```

3. Install dependencies:
   ```bash
   poetry install
   ```

4. Activate the Poetry environment:
   ```bash
   poetry shell
   ```

### Project Structure

The project is organized as follows:

```
notes-on-ml/
├── data/               # Data files
│   ├── raw/           # Raw data
│   └── processed/     # Processed data
├── docs/              # Documentation
├── notebooks/         # Jupyter notebooks
├── reports/           # Generated reports
├── src/               # Source code
│   ├── data/         # Data processing
│   ├── features/     # Feature engineering
│   ├── models/       # ML models
│   └── visualization/# Visualization
└── tests/            # Test files
```

## Using the Project

### Data Management

#### Working with Data

1. Place raw data files in `data/raw/`
2. Process data using scripts in `src/ml/data/`
3. Processed data will be saved in `data/processed/`

Example:
```python
from ml.data import process_data

# Process raw data
df = process_data('data/raw/input.csv')
```

#### Data Version Control

The project uses DVC for data versioning:

```bash
# Track data files
dvc add data/raw/input.csv

# Push changes to remote storage
dvc push

# Pull latest data
dvc pull
```

### Model Development

#### Training Models

1. Configure model parameters in `src/ml/models/config.py`
2. Train models using scripts in `src/ml/models/`
3. Track experiments with MLflow

Example:
```python
from ml.models import train_model

# Train model
model = train_model(
    data_path='data/processed/train.csv',
    model_type='random_forest'
)
```

#### Experiment Tracking

View MLflow dashboard:
```bash
poetry run mlflow ui
```

### Visualization

#### Creating Visualizations

1. Use functions in `src/ml/visualization/`
2. Save figures in `reports/figures/`

Example:
```python
from ml.visualization import plot_results

# Create visualization
plot_results(
    data=results_df,
    output_path='reports/figures/results.png'
)
```

#### Interactive Dashboards

Run Streamlit dashboard:
```bash
poetry run streamlit run src/ml/visualization/dashboard.py
```

### Jupyter Notebooks

1. Start Jupyter Lab:
   ```bash
   poetry run jupyter lab
   ```

2. Create new notebooks in `notebooks/`
3. Use project modules in notebooks:
   ```python
   import sys
   sys.path.append('..')
   from ml.data import process_data
   ```

## Common Tasks

### Running Tests

```bash
poetry run pytest
```

### Building Documentation

```bash
poetry run mkdocs build
```

### Updating Dependencies

```bash
poetry update
```

## Troubleshooting

### Common Issues

1. **Module Not Found**
   - Ensure you're in the Poetry environment
   - Check if the module is installed
   - Verify import paths

2. **Data Access Issues**
   - Check file permissions
   - Verify data paths
   - Run `dvc pull` for latest data

3. **Model Training Issues**
   - Check input data format
   - Verify model parameters
   - Check available memory

### Getting Help

- Check the documentation
- Search existing issues
- Create a new issue
- Contact the maintainers

## Best Practices

### Data Management

- Keep raw data immutable
- Document data processing steps
- Use appropriate data formats
- Version control data files

### Model Development

- Document model parameters
- Save model artifacts
- Track experiments
- Validate model performance

### Code Organization

- Follow project structure
- Use appropriate modules
- Document code changes
- Write unit tests

## Contributing

We welcome contributions! Please see the [Contributing Guide](contributing.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
