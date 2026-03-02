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

train_df = load_to_multiindex("../data/processed/imdb/movie/train.jsonl")
#valid_file = read_valid_file()

train_df[("left", "REL_SCORE")] = 0.5
train_df[("right", "REL_SCORE")] = 0.5

for idx, row in train_df.iterrows():
    left_id = row[("left", COL_TCONST)]
    right_id = row[("right", COL_TCONST)]

    authors_left = m_to_p.get(left_id, set())
    authors_right = m_to_p.get(right_id, set())

    max_pool_score = 0.0
    
    # Only compute if both sets exist
    if authors_left and authors_right:
        is_match_found = False
        for a_left in authors_left:
            for a_right in authors_right:
                try:
                    # UnionFind.find will fail if ID wasn't in basics_csv
                    # Ensure both IDs exist in the UF structure
                    if a_left in uf_p.parent and a_right in uf_p.parent:
                        if uf_p.find(a_left) == uf_p.find(a_right):
                            is_match_found = True
                            break
                except KeyError:
                    continue
            if is_match_found: break
        
        max_pool_score = 1.0 if is_match_found else 0.0
    else:
        # Default for cases where no principal data is available
        max_pool_score = 0.5

    # 2. Implement Signal Dropout
    # Randomly decide whether to keep the score or use the neutral [UNK] value (0.5)
    if random.random() < DROPOUT_PROB:
        final_score = 0.5  # Dropout / Unknown state
    else:
        final_score = max_pool_score

    # 2. Use .at or .loc with the specific index (idx) to update only the current row
    train_df.at[idx, ("left", "REL_SCORE")] = final_score
    train_df.at[idx, ("right", "REL_SCORE")] = final_score

# Now serialize the updated DataFrame
serialize_to_ditto(train_df, "rel_score.txt")