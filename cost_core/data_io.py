# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
data_io.py - Core I/O module for the defense cost-estimation library.
Provides robust loading and persistence for historical cost data.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Union

import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Schema Definitions
REQUIRED_COLUMNS = ["program", "lot", "unit_quantity", "unit_cost"]
OPTIONAL_COLUMNS = ["date", "notes", "source"]
ALL_VALID_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

COLUMN_TYPES = {
    "program": "string",
    "lot": "int64",
    "unit_quantity": "int64",
    "unit_cost": "float64",
    "date": "string",  # ISO-8601 recommended
    "notes": "string",
    "source": "string"
}

class ValidationError(Exception):
    """Raised when data does not meet schema requirements."""
    pass


def _validate_cost_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal helper to validate the cost estimation schema.
    Ensures required columns exist and enforces strict typing.
    """
    # 1. Missing Columns Check
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValidationError(f"Missing required columns: {missing}")

    # 2. Drop extraneous columns not in our definitions
    existing_valid = [col for col in df.columns if col in ALL_VALID_COLUMNS]
    df = df[existing_valid].copy()

    # 3. Enforce Types
    try:
        for col in existing_valid:
            # We use nullable types or standard numpy/pandas types
            df[col] = df[col].astype(COLUMN_TYPES[col])
    except (ValueError, TypeError) as e:
        raise ValidationError(f"Type enforcement failed: {e}")

    # 4. Logical Validation (Unit cost/quantity cannot be negative)
    if (df["unit_quantity"] < 0).any() or (df["unit_cost"] < 0).any():
        raise ValidationError("Negative values found in unit_quantity or unit_cost.")

    return df


def load_cost_csv(path: Union[str, Path]) -> pd.DataFrame:
    """
    Loads cost data from a CSV file.
    
    Args:
        path: Path to the source CSV file.
        
    Returns:
        pd.DataFrame: Validated cost data.
        
    Raises:
        FileNotFoundError: If path does not exist.
        ValidationError: If schema or data types are invalid.
    """
    path = Path(path)
    if not path.exists():
        logger.error(f"CSV file not found: {path}")
        raise FileNotFoundError(f"No file at {path}")

    logger.info(f"Loading cost data from {path}")
    try:
        df = pd.read_csv(path)
        return _validate_cost_schema(df)
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise


def save_to_sqlite(
    df: pd.DataFrame, 
    db_path: Union[str, Path], 
    table_name: str = "cost_history"
) -> None:
    """
    Persists a cost DataFrame to a local SQLite database.
    
    Args:
        df: The DataFrame to save.
        db_path: Path to the .sqlite file.
        table_name: Destination table name.
    """
    # Pre-save validation
    df = _validate_cost_schema(df)
    
    db_path = Path(db_path)
    logger.info(f"Saving data to table '{table_name}' at {db_path}")

    try:
        with sqlite3.connect(db_path) as conn:
            # We use 'replace' to ensure the table matches our current validated DF structure
            df.to_sql(table_name, conn, if_exists="replace", index=False)
        logger.info("Successfully persisted data.")
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        raise


def load_from_sqlite(
    db_path: Union[str, Path], 
    table_name: str = "cost_history"
) -> pd.DataFrame:
    """
    Reads cost data from a local SQLite database and validates it.
    
    Args:
        db_path: Path to the .sqlite file.
        table_name: Source table name.
        
    Returns:
        pd.DataFrame: Validated cost data.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    logger.info(f"Reading table '{table_name}' from {db_path}")
    try:
        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        
        return _validate_cost_schema(df)
    except Exception as e:
        logger.error(f"Failed to load from SQLite: {e}")
        raise