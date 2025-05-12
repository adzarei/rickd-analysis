"""Utility functions for the RICKD analysis project."""

import pandas as pd


def non_na_count(df, cols: list[str]) -> int:
    """Count the number of non-NA values in the columns of a dataframe."""
    return df[cols].notna().any(axis=1).sum()


def non_na_percentage(df, cols: list[str]) -> float:
    """Percentage of non-NA values in the columns of a dataframe."""
    return non_na_count(df, cols) / len(df) * 100


def condition_report(df, df_condition: pd.Series) -> None:
    """Report the number of rows that meet a condition."""
    filtered_df = df[df_condition]

    print(f"Number of subjects that meet condition: {len(filtered_df)}")
    print(f"Percentage of total rows: {(len(filtered_df) / len(df)) * 100:.2f}%")

    print("\nExample rows:")
    print(filtered_df.head().to_string())

