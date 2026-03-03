from collections import defaultdict
import csv
import json
import random
from typing import Dict, List, Set

import pandas as pd

def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)

class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x], self.rank[x] = x, 0

    def find(self, x: str) -> str:
        root = x
        while self.parent[root] != root: root = self.parent[root]
        while x != root:
            p = self.parent[x]
            self.parent[x], x = root, p
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a); self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]: self.rank[ra] += 1

def build_unionfind_with_singletons(
    basics_csv: str, dupes_csv: str, id_col: str, 
    delimiter: str = ",", has_header: bool = True
) -> UnionFind:
    uf = UnionFind()
    with open(basics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get(id_col, "").strip()
            if uid: uf.add(uid)

    with open(dupes_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header: next(reader, None)
        for row in reader:
            if len(row) >= 2:
                a, b = row[0].strip(), row[1].strip()
                if a and b: uf.union(a, b)
    return uf

def load_to_multiindex(file_path: str, columns_to_drop: List[str] = None) -> pd.DataFrame:
    """
    Loads JSONL data and converts it into a MultiIndex DataFrame (left, right, metadata).
    """
    raw_data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data.append(tuple(json.loads(line)))

    # Unzip components
    left_list, right_list, labels = zip(*raw_data)

    # Create individual DataFrames
    df_left = pd.DataFrame(left_list)
    df_right = pd.DataFrame(right_list)
    df_label = pd.DataFrame(labels, columns=['match'])

    # Build MultiIndex
    df = pd.concat([df_left, df_right, df_label], axis=1)
    columns = (
        [('left', col) for col in df_left.columns] +
        [('right', col) for col in df_right.columns] +
        [('metadata', 'match')]
    )
    df.columns = pd.MultiIndex.from_tuples(columns)

    # Drop unwanted columns across both sides
    if columns_to_drop:
        # errors='ignore' ensures it doesn't crash if a column is already missing
        df = df.drop(columns=columns_to_drop, level=1, errors='ignore')

    return df

def serialize_to_ditto(df: pd.DataFrame, output_path: str = None) -> List[str]:
    """
    Converts a MultiIndex DataFrame into Ditto serialization format.
    """
    def format_row(row):
        # Helper to format one side into COL VAL strings
        def fmt(side):
            return " ".join([f"COL {k} VAL {v}" for k, v in row[side].items() if pd.notna(v)])
        
        return f"{fmt('left')}\t{fmt('right')}\t{row['metadata', 'match']}"

    # Use apply for faster row-wise processing
    ditto_lines = df.apply(format_row, axis=1).tolist()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(ditto_lines) + "\n")
            
    return ditto_lines

def process_relationship_scores(df, m_to_p, uf_p, col_tconst, dropout_prob):
    """
    Applies Union-Find matching logic and signal dropout to a MultiIndex DataFrame.
    """
    # Initialize with neutral score
    df[("left", "REL_SCORE")] = 0.5
    df[("right", "REL_SCORE")] = 0.5

    for idx, row in df.iterrows():
        left_id = row[("left", col_tconst)]
        right_id = row[("right", col_tconst)]

        authors_left = m_to_p.get(left_id, set())
        authors_right = m_to_p.get(right_id, set())

        max_pool_score = 0.5 # Default for missing data
        
        if authors_left and authors_right:
            is_match_found = False
            for a_left in authors_left:
                for a_right in authors_right:
                    # Check if both IDs exist in the UF structure to avoid KeyErrors
                    if a_left in uf_p.parent and a_right in uf_p.parent:
                        if uf_p.find(a_left) == uf_p.find(a_right):
                            is_match_found = True
                            break
                if is_match_found: break
            
            max_pool_score = 1.0 if is_match_found else 0.0

        # Signal Dropout logic
        final_score = 0.5 if random.random() < dropout_prob else max_pool_score

        # Update specific row
        df.at[idx, ("left", "REL_SCORE")] = final_score
        df.at[idx, ("right", "REL_SCORE")] = final_score
    
    return df

PATH_RAW_PRINCIPALS = "../data/raw/imdb/title_principals.csv"
PATH_RAW_TITLE_BASICS     = "../data/raw/imdb/title_basics.csv"
PATH_RAW_TITLE_DUPS       = "../data/raw/imdb/title_basics_dups.csv"
PATH_RAW_NAME_BASICS     = "../data/raw/imdb/name_basics.csv"
PATH_RAW_NAME_DUPS       = "../data/raw/imdb/name_basics_dups.csv"
COL_TCONST     = "tconst"
COL_NCONST = "nconst"
DROPOUT_PROB = 0.15

m_to_p = build_relation_map(PATH_RAW_PRINCIPALS, COL_TCONST, COL_NCONST)
p_to_m = build_relation_map(PATH_RAW_PRINCIPALS, COL_NCONST, COL_TCONST)
uf_p = build_unionfind_with_singletons(PATH_RAW_NAME_BASICS, PATH_RAW_NAME_DUPS, COL_NCONST)
uf_m = build_unionfind_with_singletons(PATH_RAW_TITLE_BASICS, PATH_RAW_TITLE_DUPS, COL_TCONST)


# --- Execution ---
data_paths = {
    "train": "../data/processed/imdb/movie/train.jsonl",
    "valid": "../data/processed/imdb/movie/valid.jsonl",
    "test":  "../data/processed/imdb/movie/test.jsonl" # Corrected path from your snippet
}

dataframes = {}

for split, path in data_paths.items():
    print(f"Processing {split}...")
    df = load_to_multiindex(path)
    
    # Apply logic
    df = process_relationship_scores(df, m_to_p, uf_p, COL_TCONST, DROPOUT_PROB)
    drop_list = ['primaryTitle', 'originalTitle', 'cluster_id', 'block_key']
    df = df.drop(columns=drop_list, level=1, errors='ignore')
    # Serialize
    serialize_to_ditto(df, f"../data/processed/imdb/movie/ditto/{split}_movie_rel_score.txt")
    
    # Store back to original variables if needed
    dataframes[split] = df

# Optional: Re-assign to individual variables
train_df, valid_df, test_df = dataframes["train"], dataframes["valid"], dataframes["test"]
train_df.to_parquet('movie_train_rel_score.parquet')
valid_df.to_parquet('movie_valid_rel_score.parquet')
test_df.to_parquet('movie_test_rel_score.parquet')

# NAMES
data_paths = {
    "train": "../data/processed/imdb/name/train.jsonl",
    "valid": "../data/processed/imdb/name/valid.jsonl",
    "test":  "../data/processed/imdb/name/test.jsonl" 
}

dataframes = {}

for split, path in data_paths.items():
    print(f"Processing {split}...")
    df = load_to_multiindex(path)
    
    # Apply logic
    df = process_relationship_scores(df, p_to_m, uf_m, COL_NCONST, DROPOUT_PROB)
    drop_list = ['primaryName', 'cluster_id', 'block_key']
    df = df.drop(columns=drop_list, level=1, errors='ignore')
    # Serialize
    serialize_to_ditto(df, f"../data/processed/imdb/name/ditto/{split}_name_rel_score.txt")
    
    # Store back to original variables if needed
    dataframes[split] = df

# Optional: Re-assign to individual variables
train_df, valid_df, test_df = dataframes["train"], dataframes["valid"], dataframes["test"]
train_df.to_parquet('name_train_rel_score.parquet')
valid_df.to_parquet('name_valid_rel_score.parquet')
test_df.to_parquet('name_test_rel_score.parquet')

