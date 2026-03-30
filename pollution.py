from typing import Dict, List

import pandas as pd


def apply_pollution(row: pd.Series) -> pd.Series:

    return row


def pollute(fp: str) -> Dict[str, pd.DataFrame]:
    """Reads a csv file and returns n DataFrames according to their set pollution levels. Pollution is applied incrementally so "low" is a subset of "high"

    Parameters
    ----------
    fp: str
        Filepath of the csv data

    Returns
    ----------
    Dict[str, pd.DataFrame]
        Mapping the pollution Levels to the polluted dataset as a DataFrame including the original unpolluted df
    """

    df = pd.read_csv(fp)
    total_length = len(df)
    df_by_pollution = {"source": df}

    # Tuple(label, additional_percentage_to_add)
    # additional percentage to add: how many new rows are needed? Pollution depends on previous pollution.
    # Example medium: If total pollution target is 30% and previous pollution was 10%, then 10% old rows will be re-polluted and 20% new rows need to be polluted
    stages = [
        ("low", 0.1),
        ("medium", 0.2),
        ("high", 0.2)
    ]

    current_df = df.copy()
    all_polluted_indices = pd.Index([])

    for label, add_frac in stages:
        # Which rows are not polluted (yet)?
        remaining_indices = df.index.difference(all_polluted_indices)

        # Sample new rows to reach target percentage
        n_to_sample = int(total_length * add_frac)
        new_indices = pd.Series(remaining_indices).sample(
            n=n_to_sample,
            random_state=42
        ).values

        # Pollute all indices, newly sampled + old indices
        all_polluted_indices = all_polluted_indices.union(new_indices)

        # Apply pollution to all indices. This includes indices that have already been polluted
        current_df.loc[all_polluted_indices] = current_df.loc[all_polluted_indices].apply(
            apply_pollution, axis=1
        )
        # save
        df_by_pollution[label] = current_df.copy()
    return df_by_pollution


def procedural_pollution(fp):
    df = pd.read_csv(fp)
    total_length = len(df)
    df_by_pollution = {"source": df}
    # --- Low Pollution ---
    # 10%
    pollution_level = "low"
    low_frac = 0.1
    low_df = df.copy()
    low_indices = low_df.sample(frac=low_frac, random_state=42).index
    low_df.loc[low_indices] = low_df.loc[low_indices].apply(
        apply_pollution, axis=1)
    df_by_pollution[pollution_level] = low_df

    # --- Medium Pollution ---
    # Total 30%: 10% from Low + 20% new Rows
    pollution_level = "medium"
    med_frac = 0.2
    med_df = low_df.copy()

    remaining_indices = med_df.index.difference(low_indices)
    # Sample 20% of the original df length
    med_indices = pd.Series(remaining_indices).sample(
        n=int(total_length*med_frac), random_state=42).values

    all_med_indices = low_indices.union(med_indices)
    med_df.loc[all_med_indices] = med_df.loc[all_med_indices].apply(
        apply_pollution, axis=1)
    df_by_pollution[pollution_level] = med_df

    # --- High Pollution ---
    # Total 50%: 30% from Med + 20% new Rows
    pollution_level = "high"
    high_frac = 0.2
    high_df = med_df.copy()

    remaining_indices = df.index.difference(all_med_indices)
    high_indices = pd.Series(remaining_indices).sample(
        n=int(total_length*high_frac), random_state=42).values

    all_high_indices = all_med_indices.union(high_indices)
    high_df.loc[all_high_indices] = high_df.loc[all_high_indices].apply(
        apply_pollution, axis=1)
    df_by_pollution[pollution_level] = high_df

    return df_by_pollution
