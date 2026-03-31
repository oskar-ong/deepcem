import random
from typing import Dict, List

import pandas as pd

random.seed(42)


def remove_token(polluted_row: pd.Series, attr: str):
    """ Given a non-empty attribute value, remove a token. If the value is 1 token, remove a substring of size n"""
    # polluted_row = row.copy()
    val = str(polluted_row[attr])
    substring_size: int = 3

    tokenized = val.split()

    if len(tokenized) > 1:
        token_to_remove = random.choice(tokenized)
        tokenized.remove(token_to_remove)
        polluted_row[attr] = " ".join(tokenized)
    else:
        if len(val) > substring_size:
            start = random.randint(0, len(val)-substring_size)
            polluted_row[attr] = val[:start] + val[start+3:]
        else:
            polluted_row[attr] = ""

    return polluted_row


def swap_tokens(row: pd.Series, attr: str):
    """Randomly swaps two adjacent words (tokens) in the string."""
    tokens = str(row[attr]).split()

    if len(tokens) >= 2:
        idx = random.randint(0, len(tokens) - 2)
        # Standard Python swap
        tokens[idx], tokens[idx+1] = tokens[idx+1], tokens[idx]
        row[attr] = " ".join(tokens)

    return row


def remove_attribute(row: pd.Series, attr: str):
    """Simulates missing data by setting the attribute to None."""
    row[attr] = ""
    return row


def encoding_error(row: pd.Series, attr: str):
    """Simulates Mojibake (encoding issues) common in real-world messy data."""
    val = str(row[attr])

    # Common error: decoding UTF-8 as Latin-1
    try:
        row[attr] = val.encode('utf-8').decode('latin-1')
    except (UnicodeEncodeError, UnicodeDecodeError):
        # Fallback: inject a replacement character
        row[attr] = val[:-1] + ""

    return row


def add_typo(row: pd.Series, attr: str):
    """Introduces a character-level typo (swapping two adjacent characters)."""
    val = list(str(row[attr]))

    if len(val) >= 2:
        idx = random.randint(0, len(val) - 2)
        val[idx], val[idx+1] = val[idx+1], val[idx]
        row[attr] = "".join(val)

    return row


pollution_options = {
    "remove_token": remove_token,
    "swap_tokens": swap_tokens,
    "remove_attribute": remove_attribute,
    "encoding_error": encoding_error,
    "add_typo": add_typo
}


def is_empty(row: pd.Series, attr: str) -> bool:
    val = row[attr]
    # Check for NaN / None
    if pd.isna(val):
        return True
    # Check for empty or whitespace-only strings
    if isinstance(val, str) and not val.strip():
        return True
    return False


def pollute_row(row: pd.Series) -> pd.Series:
    """
    Pollutes a row. 

    Parameters
    ----------
    row: pd.Series
        Row to be polluted

    Returns
    ----------
    pd.Series
        polluted row
    """

    polluted_row = row.copy()
    pollution_option = random.choice(list(pollution_options.keys()))
    attributes = list(polluted_row.index)
    # only consider attributes, whose value is not empty
    attrs = [a for a in attributes if not is_empty(polluted_row, a)]

    # if all attributes == empty, return
    if not attrs:
        return polluted_row

    attr = random.choice(attrs)
    polluted_row = pollution_options[pollution_option](polluted_row, attr)

    return polluted_row


def pollute(fp: str, id_col: str, drop_list: List[str]) -> Dict[str, pd.DataFrame]:
    """Reads a csv file and returns n DataFrames according to their set pollution levels. Pollution is applied incrementally so "low" is a subset of "high"

    Parameters
    ----------
    fp: str
        Filepath of the csv data
    id_col: id column of the dataset

    Returns
    ----------
    Dict[str, pd.DataFrame]
        Mapping the pollution Levels to the polluted dataset as a DataFrame including the original unpolluted df
    """

    df = pd.read_csv(fp)
    df = df.set_index(id_col)
    df = df.astype(str)
    df = df.drop(columns=drop_list)

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
            pollute_row, axis=1
        )
        # save
        df_by_pollution[label] = current_df.copy()
    return df_by_pollution
