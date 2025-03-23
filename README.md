# RICKD Analysis

This project contains analysis notebooks and scripts for the RICKD project.

## Prerequisites

- Python 3.11 installed on your system
  - You can download it from [python.org](https://www.python.org/downloads/)
  - Or install it via your system's package manager

## Setup

This project uses Poetry for dependency management. To get started:

1. Install Poetry if you haven't already:
```bash
pipx install poetry && pipx ensurepath
# Or
curl -sSL https://install.python-poetry.org | python3 -
```

2. Clone this repository:
```bash
git clone <repository-url>
cd rickd-analysis
```

3. Install dependencies:
```bash
poetry install --all-groups
```

4. Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

5. Launch Jupyter Notebook:
```bash
poetry run jupyter notebook
```

## Project Structure
- `src/`: Source code and notebooks
  - `dump/`: Temporal files untracked by git
  - `suplemental_material/`: Supplemental materials provided in the RICKD bundle. See [Running Injury Clinic Kinematic Dataset](https://doi.org/10.25452/figshare.plus.24255795.v2). NOTE: MLX files have been converted to IPYNB for the easy of visualisation for users without a Matlab installation.

## Development

The project uses pre-commit hooks to ensure code quality. The following checks are performed automatically on each commit:

- Code formatting with Black
- Code linting with Ruff
- Various file checks (trailing whitespace, merge conflicts, etc.)

To manually run the pre-commit checks on all files:
```bash
pre-commit run --all-files
```

To skip pre-commit hooks (not recommended):
```bash
git commit -m "your message" --no-verify
```
