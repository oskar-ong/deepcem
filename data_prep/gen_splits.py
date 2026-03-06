from __future__ import annotations
import csv
from dataclasses import dataclass
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

# TODO: IMPORT AS ARGPASRSE
from entity_configs.imdb_hard import CONFIGS

# --- Parameters ---
SPLIT_RATIOS   = (0.7, 0.1, 0.2)  # Train, Val, Test
NEG_RATIO      = 3                # Negatives per 1 Positive
RANDOM_SEED    = 0
BLOCK_LIMIT    = 10               # Max records per block to avoid N^2 growth
SplitMode = Literal["count", "nodes"]

def build_relation_map(csv_fp: str, column1: str, column2: str, prefix1: str, prefix2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]
            # c1, c2 = f"{prefix1}{row[column1]}", f"{prefix2}{row[column2]}"
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
    title = str(row.get("primaryTitle", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_name(row: pd.Series) -> str:
    title = str(row.get("primaryName", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def generate_hard_negatives(df: pd.DataFrame, count: int) -> List[Tuple[dict, dict, int]]:
    neg_pairs = []
    block_groups = df.groupby("block_key")
    for _, group in block_groups:
        if len(neg_pairs) >= count: break
        records = group.to_dict('records')
        if len(records) > BLOCK_LIMIT:
            random.shuffle(records)
            records = records[:BLOCK_LIMIT]
        for e1, e2 in itertools.combinations(records, 2):
            if e1["cluster_id"] != e2["cluster_id"]:
                neg_pairs.append((e1, e2, 0))
                if len(neg_pairs) >= count: break
    
    while len(neg_pairs) < count:
        s1, s2 = df.sample(2).to_dict('records')
        if s1["cluster_id"] != s2["cluster_id"]:
            neg_pairs.append((s1, s2, 0))
    return neg_pairs[:count]

def generate_pairs_for_subset(subset_df: pd.DataFrame, neg_ratio: int = NEG_RATIO) -> List[Tuple[dict, dict, int]]:
    pos_pairs = []
    groups = subset_df.groupby("cluster_id")
    for _, group in groups:
        if len(group) > 1:
            for e1, e2 in itertools.combinations(group.to_dict('records'), 2):
                pos_pairs.append((e1, e2, 1))
    neg_pairs = generate_hard_negatives(subset_df, len(pos_pairs) * neg_ratio)
    return pos_pairs + neg_pairs

def propagate_dependency_pairs(
    parent_pairs: List[Tuple[dict, dict, int]], 
    dependency_map: Dict[str, Set[str],],
    id_col: str
) -> List[Tuple[str, str, int]]:

    required_name_pairs = set()
    
    for p1_dict, p2_dict, _label in parent_pairs:
        # Get IDs (e.g., 'tt12345')
        id1, id2 = p1_dict[id_col], p2_dict[id_col]
        
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

import random

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

def process_relationship_scores(df, entity_to_deps, dep_uf, main_id_col, dropout_prob, rel_name=""):
    """
    Applies Union-Find matching logic and signal dropout to a MultiIndex DataFrame.
    Dynamically names the output column to support multiple relationship types.
    """
    # Create a dynamic column name based on the entity type
    score_col = f"REL_SCORE_{rel_name.upper()}" if rel_name else "REL_SCORE"
    
    # Initialize with neutral score
    df[("left", score_col)] = 0.5
    df[("right", score_col)] = 0.5

    for idx, row in df.iterrows():
        left_id = row[("left", main_id_col)]
        right_id = row[("right", main_id_col)]

        # Get dependent entities (formerly "authors")
        deps_left = entity_to_deps.get(left_id, set())
        deps_right = entity_to_deps.get(right_id, set())

        max_pool_score = 0.5 # Default for missing data
        
        if deps_left and deps_right:
            is_match_found = False
            for d_left in deps_left:
                for d_right in deps_right:
                    # Check if both IDs exist in the UF structure to avoid KeyErrors
                    if d_left in dep_uf.parent and d_right in dep_uf.parent:
                        if dep_uf.find(d_left) == dep_uf.find(d_right):
                            is_match_found = True
                            break
                if is_match_found: break
            
            max_pool_score = 1.0 if is_match_found else 0.0

        # Signal Dropout logic
        final_score = 0.5 if random.random() < dropout_prob else max_pool_score

        # Update specific row with the dynamic column
        df.at[idx, ("left", score_col)] = final_score
        df.at[idx, ("right", score_col)] = final_score
    
    return df

def process_and_save_ditto(file_path: str, columns_to_drop: List[str], output_path: str):
    """
    Main pipeline function.
    """
    # 1. Load and Clean
    df = load_to_multiindex(file_path, columns_to_drop)
    print(f"Loaded and cleaned {len(df)} pairs.")
    
    # 2. Serialize and Save
    serialize_to_ditto(df, output_path)
    print(f"Ditto file saved to: {output_path}")
    return df

@dataclass
class EntityConfig:
    name: str                  # e.g., "movie", "name", "studio"
    id_col: str                # e.g., "tconst", "nconst"
    id_prefix: str             # e.g., "tt", "nm" (used to extract IDs from components)
    path_basics: str
    path_dups: str
    path_out_dir: str          # e.g., "./data/processed/imdb/movie/"
    block_key_func: Callable
    drop_list: List[str]
    is_main: bool = False
    ditto_dir: str = ""
    
    # Only needed for dependent entities: how they relate to the main entity
    rel_csv_path: str = ""
    rel_main_col: str = ""
    rel_dep_col: str = ""


# CONFIGS = [
#     # Create config entry for each entity

#     # MAIN entity
#     EntityConfig(
#         name="movie",
#         id_col="tconst",
#         id_prefix="tt",
#         path_basics="./data/raw/imdb/title_basics.csv",
#         path_dups="./data/raw/imdb/title_basics_dups.csv",
#         path_out_dir="./data/processed/imdb_ref/movie/",
#         block_key_func=create_block_key_movie,
#         drop_list=['primaryTitle', 'originalTitle', 'cluster_id', 'block_key'],
#         is_main=True,
#         ditto_dir= "./data/processed/imdb_ref/movie/"
#     ),

#     # DEPENDENT ENTITY 
#     EntityConfig(
#         name="name",
#         id_col="nconst",
#         id_prefix="nm",
#         path_basics="./data/raw/imdb/name_basics.csv",
#         path_dups="./data/raw/imdb/name_basics_dups.csv",
#         path_out_dir="./data/processed/imdb_ref/name/",
#         block_key_func=create_block_key_name,
#         drop_list=['primaryName', 'cluster_id', 'block_key'],
#         rel_csv_path="./data/raw/imdb/title_principals.csv",
#         rel_main_col="tconst",
#         rel_dep_col="nconst",
#         ditto_dir= "./data/processed/imdb_ref/name/"
#     ),
    # DEPENDENT ENTITY 2, .., n
    # EntityConfig(
    #     name="name",
    #     id_col="nconst",
    #     id_prefix="nm",
    #     path_basics="./data/raw/imdb/name_basics.csv",
    #     path_dups="./data/raw/imdb/name_basics_dups.csv",
    #     path_out_dir="./data/processed/imdb/name/",
    #     block_key_func=create_block_key_name,
    #     drop_list=['primaryName', 'cluster_id', 'block_key'],
    #     rel_csv_path="./data/raw/imdb/title_principals.csv",
    #     rel_main_col="tconst",
    #     rel_dep_col="nconst"
    # )

# ]

def main():

    main_cfg = next(c for c in CONFIGS if c.is_main)
    dep_cfgs = [c for c in CONFIGS if not c.is_main]

    # ==========================================
    # 1. Global Connected Components (Anti-Leakage)
    # ==========================================
    # Build a massive graph: Main <-> Dep1, Main <-> Dep2, etc.
    global_rel_map = defaultdict(set)
    relation_maps = {} # Store these for inference later
    
    for dep in dep_cfgs:
        # Main to Dependent
        m_to_d = build_relation_map(dep.rel_csv_path, dep.rel_main_col, dep.rel_dep_col, main_cfg.id_prefix, dep.id_prefix)
        relation_maps[dep.name] = m_to_d
        
        # Dependent to Main
        d_to_m = build_relation_map(dep.rel_csv_path, dep.rel_dep_col, dep.rel_main_col, dep.id_prefix, main_cfg.id_prefix)
        
        # Merge into global map
        for k, v in m_to_d.items(): global_rel_map[k].update(v)
        for k, v in d_to_m.items(): global_rel_map[k].update(v)

    components = find_connected_components(global_rel_map)
    splits = assign_components_to_splits(components)

    # ==========================================
    # 2. Generic Prep & Pair Generation
    # ==========================================
    processed_entities = {}

    for cfg in CONFIGS:
        print(f"Processing entity: {cfg.name}")
        
        # Extract IDs specific to this entity from the global splits
        def get_ids(comps):
            return {node for c in comps for node in c if node.startswith(cfg.id_prefix)}
        
        train_ids, valid_ids, test_ids = map(get_ids, splits)

        # Load & Cluster
        df_basics = pd.read_csv(cfg.path_basics)
        uf = build_unionfind_with_singletons(cfg.path_basics, cfg.path_dups, cfg.id_col)
        mapping = {entity: uf.find(entity) for entity in uf.parent.keys()}

        def prep_subset(ids):
            df = df_basics[df_basics[cfg.id_col].isin(ids)].copy()
            df['REL_SCORE'] = ""
            df['cluster_id'] = df[cfg.id_col].map(mapping)
            df['block_key'] = df.apply(cfg.block_key_func, axis=1)
            return df

        train_df, valid_df, test_df = map(prep_subset, [train_ids, valid_ids, test_ids])

        # Generate Pairs
        p_train, p_valid, p_test = map(generate_pairs_for_subset, [train_df, valid_df, test_df])

        # Store for saving and inference later
        processed_entities[cfg.name] = {
            "uf": uf,
            "df_basics": df_basics,
            "pairs": {"train": p_train, "valid": p_valid, "test": p_test}
        }

    # ==========================================
    # 3. Dependent Inference (Cartesian Product)
    # ==========================================
    main_test_pairs = processed_entities[main_cfg.name]["pairs"]["test"]

    for dep in dep_cfgs:
        print(f"Generating inference for dependent: {dep.name}")
        m_to_d_map = relation_maps[dep.name]
        
        # Create cartesian product pairs
        p_inference = propagate_dependency_pairs(main_test_pairs, m_to_d_map, main_cfg.id_col)
        
        # Label them
        labeled_inference = add_labels(
            p_inference, 
            processed_entities[dep.name]["uf"], 
            processed_entities[dep.name]["df_basics"], 
            dep.id_col
        )
        
        processed_entities[dep.name]["pairs"]["inference"] = labeled_inference

    # ==========================================
    # 4. Save to Disk & Ditto Serialization
    # ==========================================
    DROPOUT_PROB = 0.15

    for cfg in CONFIGS:
        pairs_dict = processed_entities[cfg.name]["pairs"]
        
        # Save standard splits
        for split_name in ["train", "valid", "test"]:
            out_path = f"{cfg.path_out_dir}{split_name}.jsonl"
            pairs = pairs_dict[split_name]
            random.shuffle(pairs)
            with open(out_path, "w") as f:
                for entry in pairs:
                    f.write(json.dumps(entry) + "\n")
                    
        # Save inference (only for dependents)
        if not cfg.is_main:
            inf_path = f"{cfg.path_out_dir}inference.jsonl"
            random.shuffle(pairs_dict["inference"])
            with open(inf_path, "w") as f:
                for entry in pairs_dict["inference"]:
                    f.write(json.dumps(entry) + "\n")

        template_src = f"{cfg.path_out_dir}test.jsonl" if cfg.is_main else f"{cfg.path_out_dir}inference.jsonl"
        template_dst = f"{cfg.path_out_dir}input_template.jsonl"
        write_input_json(template_src, template_dst, cfg.drop_list)

        # BASELINE 0
        baseline_0_input_json = f"{cfg.path_out_dir}baseline0/input.jsonl"
        # remove "REL_SCORE" from baseline0 experiment
        baseline_0_drop_list = cfg.drop_list.copy()
        baseline_0_drop_list.append("REL_SCORE")
        write_input_json(f"{cfg.path_out_dir}test.jsonl", baseline_0_input_json, baseline_0_drop_list)


        # --- D. Ditto Specific Serialization ---
        for split in ["train", "valid", "test"]:
            jsonl_in = f"{cfg.path_out_dir}{split}.jsonl"
            ditto_txt_out = f"{cfg.ditto_dir}{split}.txt"
            rel_score_out = f"{cfg.ditto_dir}{split}_rel_score.txt"
            
            # 1. Base Ditto string formatting
            df_ditto = process_and_save_ditto(jsonl_in, cfg.drop_list, ditto_txt_out)
            
            # 2. Relationship Scoring
            if cfg.is_main:
                # Main entity gets relationship scores injected from ALL its dependents
                for dep in dep_cfgs:
                    m_to_d_map = relation_maps[dep.name] 
                    dep_uf = processed_entities[dep.name]["uf"]
                    
                    df_ditto = process_relationship_scores(
                        df_ditto, m_to_d_map, dep_uf, cfg.id_col, DROPOUT_PROB, rel_name=dep.name
                    )
            else:
                # Dependent entity gets relationship scores injected from the Main entity
                d_to_m_map = build_relation_map(cfg.rel_csv_path, cfg.rel_dep_col, cfg.rel_main_col, dep.id_prefix, main_cfg.id_prefix)
                main_uf = processed_entities[main_cfg.name]["uf"]
                
                df_ditto = process_relationship_scores(
                    df_ditto, d_to_m_map, main_uf, cfg.id_col, DROPOUT_PROB, rel_name=main_cfg.name
                )
                
            # 3. Final serialization to text
            serialize_to_ditto(df_ditto, rel_score_out)

            ### BASELINE 0
            jsonl_in = f"{cfg.path_out_dir}{split}.jsonl"
            ditto_txt_out = f"{cfg.path_out_dir}baseline0/{split}.txt"
            df_ditto = process_and_save_ditto(jsonl_in, baseline_0_drop_list, ditto_txt_out)
            

    return processed_entities[main_cfg.name]["pairs"]["test"] # or whatever you need to return
if __name__ == "__main__":
    main()