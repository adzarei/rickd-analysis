"""Module that contains functions and abstractions that clean datasets."""

import pandas as pd


def identify_conflicting_data(df: pd.DataFrame ,key_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    """Returns a dataframe with only the rows that have different values for the target columns grouped by the key columns."""
    return df.groupby(key_cols).filter(lambda g: (g[target_cols].nunique() > 1).any()).sort_values(by=key_cols)



def replace_values(df: pd.DataFrame, old_values: list[str], new_value: str, cols: list[str] = None) -> pd.DataFrame:
    target_cols = cols if cols else df.columns
    return df[target_cols].replace(old_values, new_value)


def standardize_free_text_col(df: pd.DataFrame, cols: list[str], non_standard_separators: list[str] = None) -> pd.DataFrame:
    """Standardizes the free text column.

    Actions performed:
    - Convert to lowercase
    - Remove leading and trailing spaces
    - Replace non-standard separators with comma
    - Replace multiple spaces with single space
    - Replace multiple commas with single comma
    """
    for col in cols:
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()
        separators = non_standard_separators if non_standard_separators else r'[;&/]|(\s+and\s+)'
        df[col] = df[col].str.replace(separators, ',', regex=True)

        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        df[col] = df[col].str.replace(r',\s*', ', ', regex=True)
        df[col] = df[col].str.strip(',')

    return df


def to_lowercase(df: pd.DataFrame, cols: list[str]=None) -> pd.DataFrame:
    """Converts the specified columns to lowercase."""
    target_cols = cols if cols else df.columns
    for col in target_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    return df


def map_injury_codes(df: pd.DataFrame, mapping_file_path: str, source_col: str, target_col: str, map_key_col='key', map_value_col='value') -> pd.DataFrame:
    """Map injury names to standardized codes using a mapping file.
    
    Args:
        df: Source dataframe containing injury names
        mapping_file_path: Path to CSV file containing injury name to code mappings
        source_col: Name of column containing injury names to map
        target_col: Name of new column to store mapped codes
        
    Returns:
        DataFrame with new column containing mapped codes
    """
    # Read mapping file and create dictionary
    mapping_df = pd.read_csv(mapping_file_path)
    mapping_dict = dict(zip(mapping_df[map_key_col], mapping_df[map_value_col]))

    result_df = df.copy()
    
    # Map values using the mapping dictionary
    result_df[target_col] = result_df[source_col].map(mapping_dict)
    
    # Check for unmapped values
    unmapped = result_df[result_df[target_col].isna() & result_df[source_col].notna()]
    if not unmapped.empty:
        print(f"Warning: Found {len(unmapped)} unmapped values in {source_col}")
        print("Example unmapped values:")
        print(unmapped[[source_col]].head())
    
    return result_df