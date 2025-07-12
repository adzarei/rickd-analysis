"""Utility functions for the RICKD analysis project."""

import pandas as pd
import numpy as np
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
    Save a DataFrame as table visualization, ensuring all values (even long ones) are correctly visualized and fit in the table.
    
    Args:
        df (pd.DataFrame): DataFrame to visualize
        target_path (str): Path where the image should be saved
        figsize (tuple): Figure size as (width, height)
        fontsize (int): Font size for table text
        scale (float): Scale factor for table
        dpi (int): DPI for saved image
    """
    # Convert all values to string to ensure correct visualization (including NaN)
    cell_text = df.astype(str).values

    # Calculate optimal column widths based on the maximum string length in each column
    col_widths = []
    padding = 2  # extra chars for padding
    for i, col in enumerate(df.columns):
        max_len = max([len(str(x)) for x in df.iloc[:, i].tolist()] + [len(str(col))])
        col_widths.append(max_len + padding)

    # Normalize column widths to sum to 1 (matplotlib expects fractions)
    total = sum(col_widths)
    col_widths_norm = [w / total for w in col_widths]

    # Adjust figure size if needed to accommodate very wide tables
    min_fig_width = 2 + sum([max(2, w * 0.15) for w in col_widths])  # heuristic
    fig_width = max(figsize[0], min_fig_width)
    fig_height = figsize[1]

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Add row labels if present (index)
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        row_labels = [str(idx) for idx in df.index]
        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            rowLoc='center',
            colWidths=col_widths_norm
        )
    else:
        table = ax.table(
            cellText=cell_text,
            colLabels=df.columns,
            loc='center',
            cellLoc='center',
            colWidths=col_widths_norm
        )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, scale)

    # Make header row bold
    for i in range(len(df.columns)):
        table[(0, i)].set_text_props(weight='bold')

    # If row labels are present, make them bold as well
    if df.index.name or not isinstance(df.index, pd.RangeIndex):
        for i in range(len(df)):
            table[(i+1, -1)].set_text_props(weight='bold')

    plt.savefig(target_path, bbox_inches='tight', dpi=dpi)
    plt.close(fig)
