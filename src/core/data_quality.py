"""Module that contains functions and abstractions that clean datasets."""

import pandas as pd


def identify_conflicting_data(df: pd.DataFrame ,key_cols: list[str], target_cols: list[str]) -> pd.DataFrame:
    """Returns a dataframe with only the rows that have different values for the target columns grouped by the key columns."""
    return df.groupby(key_cols).filter(lambda g: (g[target_cols].nunique() > 1).any()).sort_values(by=key_cols)


if __name__ == "__main__":
    from core.constants import RICKD_RUNNING_METADATA_FILE
    df = pd.read_csv(RICKD_RUNNING_METADATA_FILE)
    print(identify_conflicting_data(df, ["sub_id"], ["age", "Gender", "Height", "Weight", "DominantLeg"]))
