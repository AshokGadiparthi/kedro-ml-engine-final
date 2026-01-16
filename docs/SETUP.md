# Setup Instructions

## Prerequisites
- Python 3.9+
- pip
- Git

## Installation

### 1. Extract ZIP
```bash
unzip ml-engine-phase1-complete.zip
cd ml-engine-phase1
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -e ".[dev]"
```

### 4. Verify Installation
```bash
kedro --version
pytest tests/ -v
```

## Running Pipelines

```bash
# Run all pipelines
kedro run

# Run specific pipeline
kedro run --pipeline=data_cleaning

# Visualize
kedro viz
```

## Docker

```bash
docker-compose up
```
