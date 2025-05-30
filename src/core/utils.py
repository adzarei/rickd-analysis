"""Utility functions for the RICKD analysis project."""

import pandas as pd
import matplotlib.pyplot as plt


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



def save_df_as_table_image(df, target_path, figsize=(15, 10), fontsize=9, scale=1.5, dpi=300):
    """
    Save a DataFrame as table visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing injury counts
        target_path (str): Path where the image should be saved
        figsize (tuple): Figure size as (width, height)
        fontsize (int): Font size for table text
        scale (float): Scale factor for table
        dpi (int): DPI for saved image
    """
    plt.figure(figsize=figsize)
    
    table = plt.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale)
    
    # Make header row bold
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold')
    
    plt.axis('off')
    plt.savefig(target_path, bbox_inches='tight', dpi=dpi)
    plt.close()
