# Setup Instructions - ML Engine v0.2.0

## Prerequisites

- **Python 3.9+** (tested with 3.12)
- **pip >= 23.0**
- **Git**
- **2GB RAM** minimum
- **500MB disk space**

## Installation Steps

### 1. Clone/Extract Repository

```bash
unzip ml-engine-latest.zip
cd ml-engine-latest
```

### 2. Create Virtual Environment

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Upgrade pip and setuptools

```bash
pip install --upgrade pip setuptools wheel
```

### 4. Install Dependencies

```bash
# Install with development tools
pip install -e ".[dev]"

# Or install with just core dependencies
pip install -e .
```

### 5. Verify Installation

```bash
# Check Python version
python --version

# Check Kedro version
kedro --version

# Run tests
pytest tests/ -v --cov

# Run pipeline
kedro run
```

## Expected Output

### Kedro Version Check
```bash
$ kedro --version
kedro, version 0.19.5
```

### Test Run
```bash
$ pytest tests/ -v
======================== 15 passed in 2.34s =========================
======================== coverage: 95%+ ========================
```

### Pipeline Run
```bash
$ kedro run
================================================================================
🚀 Pipeline Starting
================================================================================
📊 Loading raw data...
✅ Loaded 1000 rows, 20 columns
🔍 Validating data quality...
✅ Validation passed
🧹 Cleaning data...
✅ Cleaned data shape: (998, 20)
================================================================================
✅ Pipeline Completed Successfully
================================================================================
```

## Running Pipelines

### Run All Pipelines
```bash
kedro run
```

### Run Specific Pipeline
```bash
# Load pipeline
kedro run --pipeline=data_loading

# Validate pipeline
kedro run --pipeline=data_validation

# Clean pipeline
kedro run --pipeline=data_cleaning
```

### Visualize Pipeline
```bash
kedro viz
```

## Docker Setup

### Build Docker Image
```bash
docker build -t ml-engine:latest .
```

### Run with Docker Compose
```bash
docker-compose up
```

## Troubleshooting

### Kedro Command Not Found
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall Kedro
pip install --upgrade kedro==0.19.5
```

### Python 3.12 Compatibility Warning
All warnings are suppressed. The system is fully compatible with Python 3.12.

### Module Not Found
```bash
# Reinstall package in development mode
pip install -e ".[dev]" --force-reinstall
```

### Port Already in Use
Edit `docker-compose.yml` and change the port mapping before running.

## Next Steps

1. ✅ Verify installation works
2. ✅ Run tests successfully
3. ✅ Run pipeline successfully
4. ✅ Add your data to `data/01_raw/`
5. ✅ Update configuration as needed
6. ✅ Run with your data

## Support

Check other documentation files for detailed information:
- `docs/PHASE_1.md` - Phase 1 details
- `docs/API.md` - API reference
- `README.md` - Project overview
