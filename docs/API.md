# API Reference - ML Engine v0.2.0

## Pipelines

### data_loading_pipeline()
Loads raw data from CSV files.

**Inputs:**
- `params:data_path` - Path to CSV file

**Outputs:**
- `raw_data` - DataFrame with raw data

**Example:**
```python
from ml_engine.pipelines.data_loading import create_pipeline

pipeline = create_pipeline()
# Loads data from conf/base/parameters.yml:data_path
```

### data_validation_pipeline()
Validates data quality and generates report.

**Inputs:**
- `raw_data` - DataFrame to validate

**Outputs:**
- `data_validation_report` - Validation report (dict)

**Example:**
```python
from ml_engine.pipelines.data_validation import create_pipeline

pipeline = create_pipeline()
# Validates row count, column count, missing values
```

### data_cleaning_pipeline()
Cleans data by removing duplicates and handling missing values.

**Inputs:**
- `raw_data` - DataFrame to clean
- `params:data_processing.handle_missing` - Strategy (drop, median, mean)

**Outputs:**
- `cleaned_data` - Cleaned DataFrame
- `data_cleaning_report` - Cleaning report (dict)

**Example:**
```python
from ml_engine.pipelines.data_cleaning import create_pipeline

pipeline = create_pipeline()
# Removes duplicates and handles missing values
```

## Utilities

### validate_dataframe()

```python
def validate_dataframe(
    df: pd.DataFrame, 
    min_rows: int = 10, 
    min_cols: int = 2
) -> Dict[str, Any]:
```

Validate DataFrame structure and content.

**Parameters:**
- `df` - DataFrame to validate
- `min_rows` - Minimum required rows
- `min_cols` - Minimum required columns

**Returns:**
```python
{
    "valid": bool,
    "errors": List[str],
    "warnings": List[str],
    "stats": {
        "rows": int,
        "columns": int,
        "memory_mb": float,
    }
}
```

**Example:**
```python
from ml_engine.utils.validators import validate_dataframe

report = validate_dataframe(df)
if report["valid"]:
    print("✅ Validation passed")
else:
    print(f"❌ Errors: {report['errors']}")
```

### validate_X_y()

```python
def validate_X_y(
    X: pd.DataFrame, 
    y: pd.Series
) -> bool:
```

Validate features and target alignment.

**Parameters:**
- `X` - Features DataFrame
- `y` - Target Series

**Raises:**
- `DataValidationError` - If X and y don't match
- `InsufficientDataError` - If less than 10 samples

**Example:**
```python
from ml_engine.utils.validators import validate_X_y

validate_X_y(X, y)  # Raises if invalid
```

## Exceptions

### MLEngineException
Base exception for all ML Engine errors.

```python
from ml_engine.utils.exceptions import MLEngineException

try:
    # some operation
except MLEngineException as e:
    print(f"Error: {e}")
    print(f"Code: {e.error_code}")
```

### DataValidationError
Raised when data validation fails.

```python
from ml_engine.utils.exceptions import DataValidationError

raise DataValidationError(
    "Invalid data",
    error_code="VALIDATION_FAILED",
    details={"rows": 5}
)
```

### DataCleaningError
Raised when data cleaning fails.

### DataLoadingError
Raised when data loading fails.

### InsufficientDataError
Raised when dataset has too few samples.

## Logger

### setup_logging()

```python
from ml_engine.utils.logger import setup_logging

setup_logging(log_dir="logs", config_file="conf/base/logging.yml")
```

### get_logger()

```python
from ml_engine.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Processing started")
logger.error("Something went wrong")
```

## Configuration Files

### parameters.yml
Algorithm and data processing parameters.

```yaml
data_path: "data/01_raw/data.csv"

data_processing:
  handle_missing: "median"  # drop, median, mean
  remove_duplicates: true

algorithms:
  xgboost:
    n_estimators: 100
    learning_rate: 0.1
```

### data_catalog.yml
Data source definitions.

```yaml
raw_data:
  type: pandas.CSVDataset
  filepath: data/01_raw/data.csv

cleaned_data:
  type: pandas.CSVDataset
  filepath: data/02_intermediate/cleaned_data.csv
```

### logging.yml
Logging configuration.

```yaml
version: 1
formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    formatter: simple
    
  file:
    class: logging.handlers.RotatingFileHandler
    filename: logs/ml_engine.log
```

---

**API Documentation for ML Engine v0.2.0** 📚
