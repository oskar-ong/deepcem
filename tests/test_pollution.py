import pandas as pd

from pollution import pollute, pollute_row
fp = "./data/raw/imdb/title_basics.csv"
row = pd.Series({"name": "Apple iPhone 13",
                "category": "Electronics"})


def test_length():

    dfs = pollute(fp, "tconst")

    for name, df in dfs.items():
        df.to_csv(f"tests/out/imdb_title_polluted_{name}.csv")

    assert len(dfs) == 4


def test_pollute_row_same_index():

    polluted_row = pollute_row(row)
    print(row.index)
    print(polluted_row.index)
    assert list(polluted_row.index) == list(row.index)
