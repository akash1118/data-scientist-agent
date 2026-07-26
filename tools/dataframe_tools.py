# tools/dataframe_tools.py
# --------------------------
# WHY: The Profiler Agent needs several pandas operations (shape, missing
#      values, dtypes, summary stats...). We split these into small, testable
#      "tool" functions instead of writing one giant function. This is also
#      how LangGraph/LangChain "tools" are meant to be used - small reusable
#      building blocks that an agent calls.
# WHAT: A collection of plain functions that each take a DataFrame and return
#       a simple Python dictionary or list (easy to show in Streamlit or send to an LLM).
# HOW: Straightforward pandas calls - no advanced tricks, so beginners can
#      follow along line by line.

import pandas as pd


def get_dataframe_shape(df: pd.DataFrame) -> dict:
    """WHAT: Returns the number of rows and columns in the dataset."""
    return {"rows": df.shape[0], "columns": df.shape[1]}


def get_missing_values(df: pd.DataFrame) -> dict:
    """
    WHY: Missing values are one of the first things a data analyst checks.
    WHAT: Returns how many missing (NaN) cells exist in each column.
    HOW: pandas' isnull().sum() counts missing values per column.
    """
    missing_counts = df.isnull().sum()
    # Only keep columns that actually HAVE missing values, to keep the output short.
    missing_dict = {col: int(count) for col, count in missing_counts.items() if count > 0}
    return missing_dict


def get_duplicate_row_count(df: pd.DataFrame) -> int:
    """WHAT: Returns how many fully duplicated rows exist in the dataset."""
    return int(df.duplicated().sum())


def get_column_types(df: pd.DataFrame) -> dict:
    """
    WHY: Knowing which columns are numbers vs text vs dates helps decide
         which charts and statistics make sense.
    WHAT: Splits column names into "numeric" and "categorical" (text) buckets.
    HOW: pandas' select_dtypes() filters columns by their data type.
    """
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return {"numeric_columns": numeric_columns, "categorical_columns": categorical_columns}


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    WHY: Summary statistics (mean, min, max, std, etc.) give a fast overview
         of the numeric columns in the dataset.
    WHAT: Returns pandas' describe() output as a plain dictionary.
    HOW: df.describe() computes stats, then we round the numbers for readability.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return {}
    described = numeric_df.describe().round(2)
    return described.to_dict()


def get_correlation_matrix(df: pd.DataFrame) -> dict:
    """
    WHY: Correlation shows which numeric columns move together
         (e.g. does Experience correlate with Salary?).
    WHAT: Returns a correlation matrix as a dictionary of dictionaries.
    HOW: pandas' corr() computes pairwise Pearson correlation.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        # Correlation needs at least 2 numeric columns to be meaningful.
        return {}
    correlation = numeric_df.corr().round(2)
    return correlation.to_dict()


def get_unique_value_counts(df: pd.DataFrame, max_columns: int = 10) -> dict:
    """
    WHY: For categorical columns (like "Department"), it's useful to know
         how many different values exist and how often each one appears.
    WHAT: Returns the top value counts for each categorical column.
    HOW: pandas' value_counts() counts occurrences of each unique value.
    """
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()
    unique_values = {}
    for col in categorical_columns[:max_columns]:
        # head(5) keeps the output short - we only need a preview, not the full list.
        unique_values[col] = df[col].value_counts().head(5).to_dict()
    return unique_values


def build_full_profile(df: pd.DataFrame) -> dict:
    """
    WHY: The Profiler Agent needs ONE function that runs every profiling
         step and returns everything together as structured JSON-like data.
    WHAT: Combines all the smaller tool functions above into a single dictionary.
    HOW: Simply calls each helper function and assembles the results.
    """
    column_types = get_column_types(df)
    return {
        "shape": get_dataframe_shape(df),
        "missing_values": get_missing_values(df),
        "duplicate_rows": get_duplicate_row_count(df),
        "numeric_columns": column_types["numeric_columns"],
        "categorical_columns": column_types["categorical_columns"],
        "summary_statistics": get_summary_statistics(df),
        "correlation": get_correlation_matrix(df),
        "unique_values": get_unique_value_counts(df),
    }
