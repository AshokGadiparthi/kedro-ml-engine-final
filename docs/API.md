# API Reference

## Pipelines

### data_loading_pipeline
Loads raw data from CSV files.

### data_validation_pipeline
Validates data quality and generates report.

### data_cleaning_pipeline
Cleans data (removes duplicates, handles missing values).

## Utilities

### Validators
- `validate_dataframe()` - Validate DataFrame structure
- `validate_X_y()` - Validate features and target

### Exceptions
- `DataValidationError` - Data validation failed
- `DataCleaningError` - Data cleaning failed
- `InsufficientDataError` - Not enough samples
