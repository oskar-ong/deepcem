from __future__ import annotations
import csv
import itertools
import json
import random
import re
from collections import defaultdict
from difflib import SequenceMatcher
from itertools import combinations
from queue import Queue
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import pandas as pd

from imdb_build_rel_train import process_relationship_scores, serialize_to_ditto
from features import process_and_save_ditto

# ==========================================
# CENTRAL CONSTANTS & CONFIGURATION
# ==========================================

# --- File Paths ---
PATH_RAW_PRINCIPALS = "../data/raw/imdb/title_principals.csv"
PATH_RAW_TITLE_BASICS     = "../data/raw/imdb/title_basics.csv"
PATH_RAW_TITLE_DUPS       = "../data/raw/imdb/title_basics_dups.csv"
PATH_RAW_NAME_BASICS     = "../data/raw/imdb/name_basics.csv"
PATH_RAW_NAME_DUPS       = "../data/raw/imdb/name_basics_dups.csv"

PATH_OUT_MOVIE_TRAIN      = "../data/processed/imdb/movie/train.jsonl"
PATH_OUT_MOVIE_VALID      = "../data/processed/imdb/movie/valid.jsonl"
PATH_OUT_MOVIE_TEST       = "../data/processed/imdb/movie/test.jsonl"
PATH_OUT_MOVIE_TEST_WO_LABEL       = "../data/processed/imdb/movie/input_template.jsonl"
PATH_OUT_NAME_TRAIN      = "../data/processed/imdb/name/train.jsonl"
PATH_OUT_NAME_VALID      = "../data/processed/imdb/name/valid.jsonl"
PATH_OUT_NAME_TEST       = "../data/processed/imdb/name/test.jsonl"
PATH_OUT_NAME_INFERENCE     = "../data/processed/imdb/name/inference.jsonl"
PATH_OUT_NAME_TEST_WO_LABEL       = "../data/processed/imdb/name/input_template.jsonl"

# --- Column Names ---
COL_TCONST     = "tconst"
COL_NCONST     = "nconst"
COL_TITLE      = "primaryTitle"
COL_NAME      = "primaryName"
COL_CLUSTER_ID = "cluster_id"
COL_BLOCK_KEY  = "block_key"

# --- Parameters ---
SPLIT_RATIOS   = (0.7, 0.1, 0.2)  # Train, Val, Test
NEG_RATIO      = 3                # Negatives per 1 Positive
RANDOM_SEED    = 0
BLOCK_LIMIT    = 10               # Max records per block to avoid N^2 growth

# --- Types ---
SplitMode = Literal["count", "nodes"]

# ==========================================
# MODULE FUNCTIONS
# ==========================================

def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)

def find_connected_components(rel_map: Dict[str, Set[str]]) -> list[Set[str]]:
    used = set()
    components = []
    for node in rel_map.keys():
        if node in used: continue
        comp, queue = set(), Queue()
        queue.put(node)
        used.add(node)
        while not queue.empty():
            u = queue.get()
            comp.add(u)
            for v in rel_map.get(u, []):
                if v not in used:
                    used.add(v)
                    queue.put(v)
        components.append(comp)
    return components

def assign_components_to_splits(
    comps: Sequence[Set[str]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
    mode: SplitMode = "nodes",
) -> Tuple[List[Set[str]], List[Set[str]], List[Set[str]]]:
    if not comps: return [], [], []
    r_train, r_val, r_test = ratios
    s = sum(ratios)
    if not (0.999 <= s <= 1.001):
        r_train, r_val, r_test = (r_train / s, r_val / s, r_test / s)

    rng = random.Random(seed)
    comps_list = list(comps)
    rng.shuffle(comps_list)

    if mode == "count":
        n = len(comps_list)
        n_train, n_val = int(round(r_train * n)), int(round(r_val * n))
        return (comps_list[:n_train], 
                comps_list[n_train:n_train + n_val], 
                comps_list[n_train + n_val:])

    if mode == "nodes":
        total_nodes = sum(len(c) for c in comps_list)
        targets = {"train": r_train * total_nodes, "val": r_val * total_nodes, "test": r_test * total_nodes}
        comps_sorted = sorted(comps_list, key=len, reverse=True)
        splits = {"train": ([], 0.0), "val": ([], 0.0), "test": ([], 0.0)}

        for comp in comps_sorted:
            size = float(len(comp))
            needs = {name: (targets[name] - curr) for name, (lst, curr) in splits.items()}
            positive = {k: v for k, v in needs.items() if v > 0}
            chosen = max(positive.items(), key=lambda kv: kv[1])[0] if positive else min(splits.items(), key=lambda kv: kv[1][1])[0]
            lst, current = splits[chosen]
            lst.append(comp)
            splits[chosen] = (lst, current + size)
        return splits["train"][0], splits["val"][0], splits["test"][0]
    
    raise ValueError(f"Unknown mode={mode}")

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

def create_block_key_movie(row: pd.Series) -> str:
    title = str(row.get(COL_TITLE, "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_name(row: pd.Series) -> str:
    title = str(row.get(COL_NAME, "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def generate_hard_negatives(df: pd.DataFrame, count: int) -> List[Tuple[dict, dict, int]]:
    neg_pairs = []
    block_groups = df.groupby(COL_BLOCK_KEY)
    for _, group in block_groups:
        if len(neg_pairs) >= count: break
        records = group.to_dict('records')
        if len(records) > BLOCK_LIMIT:
            random.shuffle(records)
            records = records[:BLOCK_LIMIT]
        for e1, e2 in itertools.combinations(records, 2):
            if e1[COL_CLUSTER_ID] != e2[COL_CLUSTER_ID]:
                neg_pairs.append((e1, e2, 0))
                if len(neg_pairs) >= count: break
    
    while len(neg_pairs) < count:
        s1, s2 = df.sample(2).to_dict('records')
        if s1[COL_CLUSTER_ID] != s2[COL_CLUSTER_ID]:
            neg_pairs.append((s1, s2, 0))
    return neg_pairs[:count]

def generate_pairs_for_subset(subset_df: pd.DataFrame, neg_ratio: int = NEG_RATIO) -> List[Tuple[dict, dict, int]]:
    pos_pairs = []
    groups = subset_df.groupby(COL_CLUSTER_ID)
    for _, group in groups:
        if len(group) > 1:
            for e1, e2 in itertools.combinations(group.to_dict('records'), 2):
                pos_pairs.append((e1, e2, 1))
    neg_pairs = generate_hard_negatives(subset_df, len(pos_pairs) * neg_ratio)
    return pos_pairs + neg_pairs

def analyze_dataset_difficulty(pairs: List[Tuple[dict, dict, int]]):
    def sim(a, b): return SequenceMatcher(None, str(a), str(b)).ratio()
    neg_sims = [sim(p[0].get(COL_TITLE, ''), p[1].get(COL_TITLE, '')) for p in pairs if p[2] == 0]
    pos_sims = [sim(p[0].get(COL_TITLE, ''), p[1].get(COL_TITLE, '')) for p in pairs if p[2] == 1]
    avg_neg = sum(neg_sims) / len(neg_sims) if neg_sims else 0
    avg_pos = sum(pos_sims) / len(pos_sims) if pos_sims else 0
    print(f"--- Dataset Difficulty Report ---\nAvg Match Sim: {avg_pos:.4f}\nAvg Neg Sim:   {avg_neg:.4f}")

def propagate_dependency_pairs(
    parent_pairs: List[Tuple[dict, dict, int]], 
    dependency_map: Dict[str, Set[str]]
) -> List[Tuple[str, str, int]]:

    required_name_pairs = set()
    
    for p1_dict, p2_dict, _label in parent_pairs:
        # Get IDs (e.g., 'tt12345')
        id1, id2 = p1_dict[COL_TCONST], p2_dict[COL_TCONST]
        
        # Get related actors for both movies
        deps1 = dependency_map.get(id1, set())
        deps2 = dependency_map.get(id2, set())
        
        # Create the Cartesian Product: (n1, n3), (n2, n3)
        for n_a, n_b in itertools.product(deps1, deps2):
            if n_a == n_b:
                continue # Skip self-comparisons
                
            # Ensure canonical ordering for the set (n_small, n_large)
            pair = tuple(sorted((n_a, n_b)))
            required_name_pairs.add(pair)
            
    return list(required_name_pairs)

def add_labels(pairs, uf, df, id_col):
    labeled_pairs = []
    # Convert DF to dict for O(1) lookup
    df_tmp = df.copy()
    df_tmp['REL_SCORE'] = ""
    name_lookup = df_tmp.set_index(id_col, drop=False).to_dict('index')
    
    for n1, n2 in pairs:
        if n1 in name_lookup and n2 in name_lookup:
            label = 1 if uf.find(n1) == uf.find(n2) else 0
            labeled_pairs.append((name_lookup[n1], name_lookup[n2], label))
    return labeled_pairs

def write_input_json(input_fp, output_fp, columns_to_remove):

    with open(input_fp, 'r', encoding='utf-8') as infile, \
        open(output_fp, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            # Parse the line into a Python list
            data = json.loads(line)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for col in columns_to_remove:
                            item.pop(col, None)
            
            # Handles 'data' as a single dictionary
            elif isinstance(data, dict):
                for col in columns_to_remove:
                    data.pop(col, None)
            
            # Check if the last element is an integer (0 or 1) and remove it
            if isinstance(data[-1], int):
                data.pop()
                
            # Write the modified list back as a JSON line
            outfile.write(json.dumps(data) + '\n')

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    # 1. Connected Components
    m_to_p = build_relation_map(PATH_RAW_PRINCIPALS, COL_TCONST, COL_NCONST)
    p_to_m = build_relation_map(PATH_RAW_PRINCIPALS, COL_NCONST, COL_TCONST)
    components = find_connected_components({**m_to_p, **p_to_m})

    # 2. Split
    splits = assign_components_to_splits(components)
    
    def get_m_ids(comps):
        return {node for c in comps for node in c if node.startswith("tt")}
    def get_p_ids(comps):
        return {node for c in comps for node in c if node.startswith("nm")}

    train_ids_m, valid_ids_m, test_ids_m = map(get_m_ids, splits)
    train_ids_p, valid_ids_p, test_ids_p = map(get_p_ids, splits)

    # 3. Load & Cluster
    df_basics_m = pd.read_csv(PATH_RAW_TITLE_BASICS)
    uf_m = build_unionfind_with_singletons(PATH_RAW_TITLE_BASICS, PATH_RAW_TITLE_DUPS, COL_TCONST)
    mapping_m = {entity: uf_m.find(entity) for entity in uf_m.parent.keys()}

    df_basics_p = pd.read_csv(PATH_RAW_NAME_BASICS)
    uf_p = build_unionfind_with_singletons(PATH_RAW_NAME_BASICS, PATH_RAW_NAME_DUPS, COL_NCONST)
    mapping_p = {entity: uf_p.find(entity) for entity in uf_p.parent.keys()}

    def prep_subset(ids):
        df: pd.DataFrame = df_basics[df_basics[column].isin(ids)].copy()
        df['REL_SCORE'] = ""
        df[COL_CLUSTER_ID] = df[column].map(mapping)
        df[COL_BLOCK_KEY] = df.apply(create_block_key, axis=1)
        return df

    df_basics: pd.DataFrame = df_basics_m.copy()
    mapping = mapping_m
    create_block_key = create_block_key_movie
    column = COL_TCONST
    m_train, m_valid, m_test = map(prep_subset, [train_ids_m, valid_ids_m, test_ids_m])
    df_basics: pd.DataFrame = df_basics_p.copy()
    mapping = mapping_p
    column = COL_NCONST
    create_block_key = create_block_key_name
    n_train, n_valid, n_test = map(prep_subset, [train_ids_p, valid_ids_p, test_ids_p])

    # 4. Pairs
    p_train_m, p_valid_m, p_test_m = map(generate_pairs_for_subset, [m_train, m_valid, m_test])
    p_train_names, p_valid_names, p_test_names = map(generate_pairs_for_subset, [n_train, n_valid, n_test])
    p_inference_names = propagate_dependency_pairs(p_test_m, m_to_p)
    labeled_inference_names = add_labels(p_inference_names, uf_p, df_basics_p, COL_NCONST)
    # if labeled_train_names:
    #     example_pair = labeled_train_names[0]
    #     print(f"DEBUG: Does entity 1 have nconst? {'nconst' in example_pair[0]}")
    #     print(f"DEBUG: Entity keys: {example_pair[0].keys()}")
    # p_valid_names = propagate_dependency_pairs(p_valid_m, m_to_p)
    # labeled_valid_names = add_labels(p_valid_names, uf_p, df_basics_p, COL_NCONST)
    # p_test_names = propagate_dependency_pairs(p_test_m, m_to_p)
    # labeled_test_names = add_labels(p_test_names, uf_p, df_basics_p, COL_NCONST)
    # p_train_n, p_valid_n, p_test_n = map(generate_pairs_for_subset, [n_train, n_valid, n_test])


    analyze_dataset_difficulty(p_train_m)
    
    # 5. Save
    for p, path in zip([p_train_m, p_valid_m, p_test_m], [PATH_OUT_MOVIE_TRAIN, PATH_OUT_MOVIE_VALID, PATH_OUT_MOVIE_TEST]):
        random.shuffle(p)
        with open(path, "w") as f:
            for entry in p:
                f.write(json.dumps(entry) + "\n")

    for p, path in zip([p_train_names, p_valid_names, p_test_names], [PATH_OUT_NAME_TRAIN, PATH_OUT_NAME_VALID, PATH_OUT_NAME_TEST]):
        random.shuffle(p)
        with open(path, "w") as f:
            for entry in p:
                f.write(json.dumps(entry) + "\n")

    random.shuffle(p)
    with open(PATH_OUT_NAME_INFERENCE, "w") as f:
        for entry in labeled_inference_names:
            f.write(json.dumps(entry) + "\n")
    
    DROPOUT_PROB = 0.15
    drop_list = ['primaryTitle', 'originalTitle', 'cluster_id', 'block_key']
    write_input_json(PATH_OUT_MOVIE_TEST, PATH_OUT_MOVIE_TEST_WO_LABEL, drop_list)
    for ip, op, split in zip([PATH_OUT_MOVIE_TRAIN, PATH_OUT_MOVIE_VALID, PATH_OUT_MOVIE_TEST], 
                      ["../data/processed/imdb/movie/ditto/train.txt", "../data/processed/imdb/movie/ditto/valid.txt", 
                       "../data/processed/imdb/movie/ditto/test.txt"], ["train", "valid", "test"]):
        df = process_and_save_ditto(ip, drop_list, op)
        df = process_relationship_scores(df, m_to_p, uf_p, COL_TCONST, DROPOUT_PROB)
        serialize_to_ditto(df, f"../data/processed/imdb/movie/ditto/{split}_rel_score.txt")

    drop_list = ['primaryName', 'cluster_id', 'block_key']

    write_input_json(PATH_OUT_NAME_INFERENCE, PATH_OUT_NAME_TEST_WO_LABEL, drop_list)
    for ip, op, split in zip([PATH_OUT_NAME_TRAIN, PATH_OUT_NAME_VALID, PATH_OUT_NAME_TEST], 
                      ["../data/processed/imdb/name/ditto/train.txt", "../data/processed/imdb/name/ditto/valid.txt", 
                       "../data/processed/imdb/name/ditto/test.txt"],["train", "valid", "test"]):
        df = process_and_save_ditto(ip, drop_list, op)
        df = process_relationship_scores(df, p_to_m, uf_m, COL_NCONST, DROPOUT_PROB)
        serialize_to_ditto(df, f"../data/processed/imdb/name/ditto/{split}_rel_score.txt")

    return p_test_m, p_test_m, p_test_m

if __name__ == "__main__":
    main()