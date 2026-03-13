from __future__ import annotations
import argparse
import csv
import itertools
import json
import pickle
import random
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Literal, Sequence, Set, Tuple

import pandas as pd

from entity_configs.entityConfig import REGISTRY

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
            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)

def find_connected_components(rel_map: Dict[str, Set[str]]) -> list[Set[str]]:
    used = set()
    components = []
    # for every entity in relation map
    for node in rel_map.keys():
        # disregard already seen entities
        if node in used: continue
        # component = set of entities
        comp = set()
        # use queue object to store nodes to look at 
        queue = Queue()
        # add entity to queue and mark as seen
        queue.put(node)
        used.add(node)
        # go through the queue
        while not queue.empty():
            u = queue.get()
            # add entity to component
            comp.add(u)
            # for every related entity to current queue pop
            for v in rel_map.get(u, []):
                # check if already seen
                if v not in used:
                    # add to queue and mark as seen
                    used.add(v)
                    queue.put(v)
        # return the component (set of nodes)
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
        # how many nodes we have in total 
        total_nodes = sum(len(c) for c in comps_list)
        # assign absolute ratios to splits
        targets = {"train": r_train * total_nodes, "val": r_val * total_nodes, "test": r_test * total_nodes}
        # sort components by number of nodes
        comps_sorted = sorted(comps_list, key=len, reverse=True)
        # split: (components, current number of nodes)
        splits = {"train": ([], 0.0), "val": ([], 0.0), "test": ([], 0.0)}

        # iterate over all components
        for comp in comps_sorted:
            # current component size (count member nodes)
            size = float(len(comp))
            # calculate how many nodes are still needed for each split
            needs = {name: (targets[name] - curr) for name, (lst, curr) in splits.items()}
            # new dict for splits that still need nodes 
            # example: {train: 100, test :30}
            positive = {k: v for k, v in needs.items() if v > 0}
            # choose the split which is missing the most nodes. If no splits need new nodes, choose the split with the lowest amount of nodes
            # positive.items() returns list of tuples (train, 100), (test:30)
            # max(key=) specifies which maximum we want
            # lambda kv: kv[1] -> function: take the 1st element (the amount of nodes needed) of the tuple as key
            # return 0 element (the split name) of tuple
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

def serialize_to_ditto(df: pd.DataFrame, output_path: str, id_col: str) -> List[str]:
    """
    Converts a MultiIndex DataFrame into Ditto serialization format.
    """
    def format_row(row):
        # Helper to format one side into COL VAL strings
        def fmt(side):
            return " ".join([f"COL {k} VAL {v}" for k, v in row[side].items() if pd.notna(v) and k != id_col])
        
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

def process_and_save_ditto(file_path: str, columns_to_drop: List[str], output_path: str, id_col):

    df = load_to_multiindex(file_path, columns_to_drop)
    print(f"Loaded and cleaned {len(df)} pairs.")
    
    serialize_to_ditto(df, output_path, id_col)
    print(f"Ditto file saved to: {output_path}")
    return df

def process_and_save_flattened_ditto(file_path: str, columns_to_drop: List[str], output_path: str, representatives: Dict, id_col):

    df = load_flattened_to_multiindex(file_path, columns_to_drop, representatives)
    print(f"Loaded and cleaned {len(df)} pairs.")
    
    # 2. Serialize and Save
    serialize_flattened_to_ditto(df, output_path, id_col)
    print(f"Ditto file saved to: {output_path}")
    return df

def serialize_flattened_to_ditto(df: pd.DataFrame, output_path: str, id_col) -> List[str]:
    def format_row(row):
        def fmt(side):
            # Iterates through columns for 'left' or 'right'
            # Skips NaN values to keep the Ditto string clean
            items = []
            for col, val in row[side].items():
                if pd.notna(val) and val != "" and col != id_col:
                    items.append(f"COL {col} VAL {val}")
            return " ".join(items)
        
        # Ditto format: LeftEntity \t RightEntity \t Label
        return f"{fmt('left')}\t{fmt('right')}\t{row['metadata', 'match']}"

    ditto_lines = df.apply(format_row, axis=1).tolist()

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(ditto_lines) + "\n")
            
    return ditto_lines

def load_flattened_to_multiindex(
    file_path: str, 
    columns_to_drop: List[str], 
    dep_map: Dict[str, str]
) -> pd.DataFrame:
    rows = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            # Structure: [left_dict, right_dict, label]
            left_ent, right_ent, label = data[0], data[1], data[2]
            
            def clean_entity(ent):
                # 1. Handle mapped dependencies (e.g., name -> primaryName)
                for dep_col, inner_key in dep_map.items():
                    if dep_col in ent and isinstance(ent[dep_col], list):
                        vals = [str(item.get(inner_key, '')) for item in ent[dep_col]]
                        ent[dep_col] = " ".join(filter(None, vals))
                
                # 2. Safety Catch: Ensure NO lists remain in any column
                # This prevents the ValueError during serialization
                for col in list(ent.keys()):
                    if isinstance(ent[col], list):
                        # Fallback: just stringify the list if not in dep_map
                        ent[col] = " ".join(map(str, ent[col]))
                    
                    if col in columns_to_drop:
                        ent.pop(col, None)
                return ent

            rows.append({
                'left': clean_entity(left_ent.copy()),
                'right': clean_entity(right_ent.copy()),
                'label': label
            })

    df_left = pd.DataFrame([r['left'] for r in rows])
    df_right = pd.DataFrame([r['right'] for r in rows])
    df_meta = pd.DataFrame([r['label'] for r in rows], columns=['match'])

    return pd.concat([df_left, df_right, df_meta], axis=1, keys=['left', 'right', 'metadata'])

def profile_components(components, configs, entity_ufs):
    analysis_data = []

    for i, comp in enumerate(components):
        row = {"comp_id": i, "total_nodes": len(comp)}
        
        for cfg in configs:
            # 1. Filter nodes of this specific type
            nodes = [n for n in comp if n.startswith(cfg.id_prefix)]
            row[f"{cfg.name}_nodes"] = len(nodes)
            
            # 2. Identify Duplicates (Unique Clusters vs. Total Nodes)
            if cfg.name in entity_ufs:
                # How many real-world entities do these nodes represent?
                #unique_clusters = {entity_ufs[cfg.name].find(n) for n in nodes}
                uf = entity_ufs[cfg.name]
                unique_clusters = {
                    uf.find(n) if n in uf.parent else n 
                    for n in nodes
                }
                row[f"{cfg.name}_clusters"] = len(unique_clusters)
                row[f"{cfg.name}_dupe_count"] = len(nodes) - len(unique_clusters)
        
        analysis_data.append(row)

    return pd.DataFrame(analysis_data)

# Usage in main():
# df_stats = profile_components(components, CONFIGS, entity_ufs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    CONFIGS = REGISTRY[args.dataset]

    main_cfg = next(c for c in CONFIGS if c.is_main)
    dep_cfgs = [c for c in CONFIGS if not c.is_main]

    # ==========================================
    # 1. Global Connected Components (Anti-Leakage)
    # ==========================================
    # Build a massive graph: Main <-> Dep1, Main <-> Dep2, etc.
    global_rel_map = defaultdict(set)
    entity_ufs = {}

    for cfg in CONFIGS:
        # create a union find for each entity type based on duplicate csv (transitive closure)
        uf = build_unionfind_with_singletons(cfg.path_basics, cfg.path_dups, cfg.id_col)
        entity_ufs[cfg.name] = uf

        # connect each duplicate to their root in global map
        # for every reference 
        for node in uf.parent.keys():
            # find the root 
            root = uf.find(node)
            # if reference is not the root -> duplicate
            if node != root:
                # add connection to global relation map
                # duplicate -> root
                global_rel_map[node].add(root)
                # root -> duplicate
                global_rel_map[root].add(node)
    
    relation_maps = {} # Store these for inference later

    for dep in dep_cfgs:
        # Main to Dependent
        m_to_d = build_relation_map(dep.rel_csv_path, dep.rel_main_col, dep.rel_dep_col, main_cfg.id_prefix, dep.id_prefix)
        relation_maps[dep.name] = m_to_d

        for m_id, d_ids in m_to_d.items():
            for d_id in d_ids:
                global_rel_map[m_id].add(d_id)
                global_rel_map[d_id].add(m_id)
        
        # Dependent to Main
        # Do i need this?
        #d_to_m = build_relation_map(dep.rel_csv_path, dep.rel_dep_col, dep.rel_main_col, dep.id_prefix, main_cfg.id_prefix)
        
        # # Merge into global map
        # # for every main, {dep_1, ..., dep_n} -> update global_rel_map for main
        # for k, v in m_to_d.items(): global_rel_map[k].update(v)
        # # for every dep, {main1, ..., main_n} -> update global_rel_map for dep
        # for k, v in d_to_m.items(): global_rel_map[k].update(v)

    components = find_connected_components(global_rel_map)

    df_stats = profile_components(components, CONFIGS, entity_ufs)
    with open ('pickles/stats.pickle', 'wb') as f:
        pickle.dump(df_stats, f, pickle.HIGHEST_PROTOCOL)


    splits = assign_components_to_splits(components)

    # pickling to evaluate created components and splits in different file
    with open ('pickles/components.pickle', 'wb') as f:
        pickle.dump(components, f, pickle.HIGHEST_PROTOCOL)
    with open ('pickles/splits.pickle', 'wb') as f:
        pickle.dump(splits, f, pickle.HIGHEST_PROTOCOL)

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
    # Create Flattened Schema (BASELINE 1)
    # ==========================================

    def get_related_records(ids, source_df, id_col):
        """Filters the source_df for the given IDs and returns records as a list of dicts."""
        if not ids:
            return []
        # Fetch all rows where the ID matches
        subset = source_df[source_df[id_col].isin(ids)]
        # Convert those rows to a list of dictionaries
        return subset.to_dict(orient='records')
    
    flattened_relations = {}
    df_main_flat = processed_entities[main_cfg.name]["df_basics"].copy()

    for dep in dep_cfgs:

        df_main_flat[dep.name] = df_main_flat[main_cfg.id_col].apply(
            lambda mid: get_related_records(relation_maps[dep.name].get(mid, []), processed_entities[dep.name]["df_basics"], dep.id_col)
        )

    master_flat_map = df_main_flat.set_index(main_cfg.id_col).to_dict(orient='index')

    def enrich_pairs_flattened(pairs: List[Tuple[dict, dict, int]], lookup_map, id_col):
        enriched_list = []
        
        for left, right, label in pairs:
            id_a = left.get(id_col)
            id_b = right.get(id_col)
            
            # Pull the full flattened attributes from our master map
            # We use .get() to avoid KeyErrors if an ID is missing
            extra_attrs_a = lookup_map.get(id_a, {})
            extra_attrs_b = lookup_map.get(id_b, {})
            
            # Merge the new attributes into the existing dictionaries
            # .copy() ensures we don't accidentally modify the original list in-place
            new_left = {**left, **extra_attrs_a}
            new_right = {**right, **extra_attrs_b}
            
            enriched_list.append((new_left, new_right, label))
            
        return enriched_list

    # ==========================================
    # Save to Disk & Ditto Serialization
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

            ## Baseline 1:
            if cfg.is_main:
                enriched_pairs = enrich_pairs_flattened(processed_entities[main_cfg.name]["pairs"][split_name], master_flat_map, cfg.id_col)
                out_path = f"{cfg.path_out_dir}/baseline1/{split_name}.jsonl"
                with open(out_path, "w") as f:
                    for entry in enriched_pairs:
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


        # BASELINE 1
        if cfg.is_main:
            baseline_1_input_json = f"{cfg.path_out_dir}baseline1/input.jsonl"
            # remove "REL_SCORE" from baseline0 experiment
            baseline_1_drop_list = cfg.drop_list.copy()
            baseline_1_drop_list.append("REL_SCORE")
            baseline_1_drop_list.append(cfg.id_col)
            write_input_json(f"{cfg.path_out_dir}baseline1/test.jsonl", baseline_1_input_json, baseline_1_drop_list)

        reps = {}
        for dep_cfg in dep_cfgs:
            reps[dep_cfg.name] = dep_cfg.rep

        # --- D. Ditto Specific Serialization ---
        for split in ["train", "valid", "test"]:
            jsonl_in = f"{cfg.path_out_dir}{split}.jsonl"
            ditto_txt_out = f"{cfg.ditto_dir}{split}.txt"
            rel_score_out = f"{cfg.ditto_dir}{split}_rel_score.txt"
            
            # 1. Base Ditto string formatting
            df_ditto = process_and_save_ditto(jsonl_in, cfg.drop_list, ditto_txt_out, cfg.id_col)

            if cfg.is_main:
                # Baseline 1 
                jsonl_in_flat = f"{cfg.path_out_dir}baseline1/{split}.jsonl"
                ditto_txt_flat_out = f"{cfg.ditto_dir}baseline1/{split}.txt"
                
                df_ditto_flattened = process_and_save_flattened_ditto(jsonl_in_flat, baseline_0_drop_list, ditto_txt_flat_out, reps, cfg.id_col)
            
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
            serialize_to_ditto(df_ditto, rel_score_out, cfg.id_col)

            ### BASELINE 0
            jsonl_in = f"{cfg.path_out_dir}{split}.jsonl"
            ditto_txt_out = f"{cfg.path_out_dir}baseline0/{split}.txt"
            df_ditto = process_and_save_ditto(jsonl_in, baseline_0_drop_list, ditto_txt_out, cfg.id_col)
            

    return processed_entities[main_cfg.name]["pairs"]["test"] # or whatever you need to return
if __name__ == "__main__":
    main()