from __future__ import annotations
from sklearn.model_selection import train_test_split
import numpy as np
import json
import argparse
import csv
import itertools
from pathlib import Path
import pickle
import random
from collections import defaultdict
from queue import Queue
import shutil
from typing import Dict, List, Literal, Set, Tuple

import pandas as pd
import networkx as nx

from data_structures import UnionFind, processedEntity
from serializer import write_splits
from entity_config import REGISTRY, EntityConfig
from pollution import pollute

# --- Parameters ---
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # Train, Val, Test
NEG_RATIO = 3                # Negatives per 1 Positive
RANDOM_SEED = 0
BLOCK_LIMIT = 10               # Max records per block to avoid N^2 growth
SplitMode = Literal["count", "nodes"]


def get_log_subsets(data, labels, base=10, min_pow=2, max_pow=5):
    """
    Generates subsets of data at sizes: base^min_pow, base^(min_pow+1)...
    Example: 100, 1000, 10000
    """
    subsets = {}

    # Calculate the integer sizes we want
    # Use np.logspace for more granular control if needed
    sizes = [base**i for i in range(min_pow, max_pow + 1)]

    for size in sizes:
        if size > len(data):
            print(
                f"Requested size {size} exceeds data length {len(data)}. Skipping.")
            continue

        # Stratify ensures the 0/1 distribution is identical to the original
        subset, _ = train_test_split(
            data,
            train_size=size,
            stratify=labels,
            random_state=42,  # Crucial for fair comparison
            shuffle=True
        )
        subsets[size] = subset
        print(f"Generated subset of size {size}")

    return subsets


def print_overlap_table(processed_entities: Dict[str, processedEntity]):
    rows = []
    for ent_name, proc in processed_entities.items():
        train_ids = set([p[0]
                        for p in proc.pairs['train']])
        test_ids = set([p[0]
                       for p in proc.pairs['test']])

        overlap = train_ids.intersection(test_ids)
        rows.append({
            "Entity": ent_name,
            "Train Nodes": len(train_ids),
            "Test Nodes": len(test_ids),
            "Overlap (Leakage)": len(overlap)
        })

    df_leakage = pd.DataFrame(rows)
    print(df_leakage.to_markdown())


def generate_metadata(args, components, processed_entities, output_path):
    metadata = {
        "dataset": args.dataset,
        "components": [],
        "splits_summary": {
            "train": {"positive_pairs": 0, "total_pairs": 0},
            "valid": {"positive_pairs": 0, "total_pairs": 0},
            "test": {"positive_pairs": 0, "total_pairs": 0}
        }
    }

    # 1. Component Metadata
    for i, comp in enumerate(components):
        # Calculate makeup: how many of each entity type
        makeup = {ent_type: len(nodes) for ent_type,
                  nodes in comp.items() if ent_type != "all_nodes"}

        comp_info = {
            "component_id": i,
            "total_size": len(comp["all_nodes"]),
            "makeup": makeup
        }
        metadata["components"].append(comp_info)

    # 2. Split Metadata (Aggregated across all entity types)
    for entity_name, proc_ent in processed_entities.items():
        for split_name in ["train", "valid", "test"]:
            pairs = proc_ent.pairs.get(split_name, [])
            pos_count = sum(1 for p in pairs if p[2] == 1)

            metadata["splits_summary"][split_name]["positive_pairs"] += pos_count
            metadata["splits_summary"][split_name]["total_pairs"] += len(pairs)

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata file created at: {output_path}")


def validate_splits(splits, global_rel_map, entity_ufs):
    report = []

    # Map node -> split_name
    node_to_split = {}
    for name, components in zip(["train", "valid", "test"], splits):
        for comp in components:
            for node_id in comp.get("all_nodes", []):
                node_to_split[node_id] = name

    # --- Leakage Check ---
    leaky_edges = 0
    for u, neighbors in global_rel_map.items():
        u_id = u[0]
        for v_id, v_type in neighbors:
            if u_id in node_to_split and v_id in node_to_split:
                if node_to_split[u_id] != node_to_split[v_id]:
                    leaky_edges += 1

    # --- Duplicate Integrity Check ---
    spilled_clusters = 0
    for ent_type, uf in entity_ufs.items():
        cluster_to_splits = defaultdict(set)
        for node_id, split in node_to_split.items():
            # Only check nodes of the current entity type
            if node_id in uf.parent:
                root = uf.find(node_id)
                cluster_to_splits[root].add(split)

        for root, split_set in cluster_to_splits.items():
            if len(split_set) > 1:
                spilled_clusters += 1

    # --- Density Check ---
    density = {}
    for name in ["train", "valid", "test"]:
        nodes_in_split = [n for n, s in node_to_split.items() if s == name]
        density[name] = len(nodes_in_split)

    print("--- DATASET HEALTH REPORT ---")
    print(f"Relational Leakage (Cross-Split Edges): {leaky_edges}")
    print(f"Duplicate Spillage (Clusters in >1 split): {spilled_clusters}")
    print(f"Node Distribution: {density}")

    return leaky_edges == 0 and spilled_clusters == 0


def build_relation_map(csv_fp: str, column1: str, column2: str, blacklist: set = None) -> Dict[str, Set[str]]:
    relation_map: Dict[str, Set[str]] = defaultdict(set)
    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c1, c2 = row[column1], row[column2]

            # disregard central nodes
            if c1 in blacklist or c2 in blacklist:
                continue

            if c1 and c2:
                relation_map[c1].add(c2)
    return dict(relation_map)


def get_dynamic_blacklist(junction_table: str, id_col: str, percentile: float = 0.95):
    """
    Identifies 'Universal Connectors' (e.g., the move 'Tackle') 
    based on the top Nth percentile of connectivity.
    """
    df = pd.read_csv(junction_table)
    counts = df[id_col].value_counts()
    # Find the threshold at the 95th percentile (the most 'hubby' nodes)
    threshold = counts.quantile(percentile)
    hubs = counts[counts > threshold].index.tolist()
    return set(hubs)


def find_connected_components(rel_map: Dict[str, Set[str]]) -> list[Dict[str, Set[str]]]:
    used = set()
    components = []
    # for every entity in relation map
    for node in rel_map.keys():
        # disregard already seen entities
        if node in used:
            continue
        # component = dict of set of entities, 1 dict entry for each entity type
        comp: Dict[str, Set[str]] = defaultdict(set)
        # use queue object to store nodes to look at
        queue = Queue()
        # add entity to queue and mark as seen
        queue.put(node)
        used.add(node)
        # go through the queue
        while not queue.empty():
            u = queue.get()
            # add entity to component
            # second element of the tuple denotes entity type
            ent_comp = comp[u[1]]
            ent_comp.add(u[0])  # add node to its entity type set
            comp["all_nodes"].add(u[0])
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
    comps: list[Dict[str, Set[str]]],
    ratios: Tuple[float, float, float] = SPLIT_RATIOS,
    seed: int = RANDOM_SEED,
    mode: SplitMode = "nodes",
) -> Tuple[list[Dict[str, Set[str]]], list[Dict[str, Set[str]]], list[Dict[str, Set[str]]]]:
    if not comps:
        return [], [], []
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
        total_nodes = sum(len(c["all_nodes"]) for c in comps_list)
        # assign absolute ratios to splits
        targets = {"train": r_train * total_nodes, "val": r_val *
                   total_nodes, "test": r_test * total_nodes}
        # sort components by number of nodes
        comps_sorted = sorted(comps_list, key=lambda kv: len(
            kv["all_nodes"]), reverse=True)
        # split: (components, current number of nodes)
        splits = {"train": ([], 0.0), "val": ([], 0.0), "test": ([], 0.0)}

        # iterate over all components
        for comp in comps_sorted:
            all_nodes = comp["all_nodes"]
            # current component size (count member nodes)
            size = float(len(all_nodes))
            # calculate how many nodes are still needed for each split
            needs = {name: (targets[name] - curr)
                     for name, (lst, curr) in splits.items()}
            # new dict for splits that still need nodes
            # example: {train: 100, test :30}
            positive = {k: v for k, v in needs.items() if v > 0}
            # choose the split which is missing the most nodes. If no splits need new nodes, choose the split with the lowest amount of nodes
            # positive.items() returns list of tuples (train, 100), (test:30)
            # max(key=) specifies which maximum we want
            # lambda kv: kv[1] -> function: take the 1st element (the amount of nodes needed) of the tuple as key
            # return 0 element (the split name) of tuple
            chosen = max(positive.items(), key=lambda kv: kv[1])[
                0] if positive else min(splits.items(), key=lambda kv: kv[1][1])[0]
            lst, current = splits[chosen]
            lst.append(comp)
            splits[chosen] = (lst, current + size)
        return splits["train"][0], splits["val"][0], splits["test"][0]

    raise ValueError(f"Unknown mode={mode}")


def build_unionfind_with_singletons(
    basics_csv: str, dupes_csv: str, id_col: str,
    delimiter: str = ",", has_header: bool = True
) -> UnionFind:
    uf = UnionFind()
    with open(basics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row.get(id_col, "").strip()
            if uid:
                uf.add(uid)

    with open(dupes_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) >= 2:
                a, b = row[0].strip(), row[1].strip()
                if a and b:
                    uf.union(a, b)
    return uf


def generate_hard_negatives(df: pd.DataFrame, count: int) -> List[Tuple[str, str, int]]:
    neg_pairs: List[str, str, int] = []
    block_groups = df.groupby("block_key")

    for _, group in block_groups:
        if len(neg_pairs) >= count:
            break
        ids = group.index.tolist()
        cluster_ids = group["cluster_id"].to_dict()  # {id: cluster_id}

        if len(ids) > BLOCK_LIMIT:
            random.shuffle(ids)
            ids = ids[:BLOCK_LIMIT]

        for id1, id2 in itertools.combinations(ids, 2):
            if cluster_ids[id1] != cluster_ids[id2]:
                neg_pairs.append((id1, id2, 0))
                if len(neg_pairs) >= count:
                    break

    while len(neg_pairs) < count:
        samples = df.sample(2)
        id1, id2 = samples.index
        c1, c2 = samples["cluster_id"]

        if c1 != c2:
            neg_pairs.append((id1, id2, 0))
    return neg_pairs[:count]


def generate_pairs_for_subset(subset_df: pd.DataFrame, neg_ratio: int = NEG_RATIO) -> List[Tuple[str, str, int]]:
    pos_pairs = []
    groups = subset_df.groupby("cluster_id")
    for _, group in groups:
        if len(group) > 1:
            # get ids grouped by index
            ids = group.index.tolist()
            for id1, id2 in itertools.combinations(ids, 2):
                pos_pairs.append((id1, id2, 1))
    neg_pairs = generate_hard_negatives(subset_df, len(pos_pairs) * neg_ratio)
    return pos_pairs + neg_pairs


def propagate_dependency_pairs(
    parent_pairs: List[Tuple[str, str, int]],
    dependency_map: Dict[str, Set[str],]
) -> List[Tuple[str, str, int]]:

    required_name_pairs = set()

    for p1, p2, _label in parent_pairs:
        id1, id2 = p1, p2

        # Get related ids for each id
        deps1 = dependency_map.get(id1, set())
        deps2 = dependency_map.get(id2, set())

        # Create the Cartesian Product: (n1, n3), (n2, n3)
        for n_a, n_b in itertools.product(deps1, deps2):
            if n_a == n_b:
                continue  # Skip self-comparisons

            # Ensure canonical ordering for the set (n_small, n_large)
            pair = tuple(sorted((n_a, n_b)))
            required_name_pairs.add(pair)

    return list(required_name_pairs)


def add_labels(pairs, uf, df, id_col):
    labeled_pairs = []
    # Convert DF to dict for O(1) lookup
    df_tmp = df.copy()
    name_lookup = df_tmp.set_index(id_col, drop=False).to_dict('index')

    for n1, n2 in pairs:
        if n1 in name_lookup and n2 in name_lookup:
            if uf.find(n1) == uf.find(n2):
                label = 1
            else:
                label = 0
            labeled_pairs.append((n1, n2, label))
    return labeled_pairs


def profile_components(components: list[Dict[str, Set[str]]], configs: Dict[str, EntityConfig], entity_ufs: Dict[str, UnionFind]):
    analysis_data = []

    for i, comp in enumerate(components):
        row = {"comp_id": i, "type_count": len(comp)}
        total_nodes = 0
        for ent_type, nodes in comp.items():
            total_nodes += len(nodes)

            row[f"{ent_type}_nodes"] = len(nodes)
            # 2. Identify Duplicates (Unique Clusters vs. Total Nodes)
            if ent_type in entity_ufs:
                # How many real-world entities do these nodes represent?
                # unique_clusters = {entity_ufs[cfg.name].find(n) for n in nodes}
                uf = entity_ufs[ent_type]
                unique_clusters = {
                    uf.find(n) if n in uf.parent else n
                    for n in nodes
                }
                row[f"{ent_type}_clusters"] = len(unique_clusters)
                row[f"{ent_type}_dupe_count"] = len(
                    nodes) - len(unique_clusters)

        analysis_data.append(row)

    return pd.DataFrame(analysis_data)


def analyze_graph_centrality(global_rel_map):
    G = nx.Graph()
    for node, neighbors in global_rel_map.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    print(
        f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # These are nodes that, if removed, increase the number of connected components.
    articulation_points = list(nx.articulation_points(G))

    # This measures how often a node appears on the shortest path between any two other nodes
    # We sample (k=...) for speed if the graph is very large.
    print("Calculating betweenness centrality (this may take a while)...")
    centrality = nx.betweenness_centrality(G, k=min(1000, len(G)//10))

    # 4. Compile Results
    analysis = []
    for node in G.nodes():
        analysis.append({
            "node_id": node[0],
            "entity_type": node[1],
            "degree": G.degree(node),
            "betweenness": centrality.get(node, 0),
            "is_articulation_point": node in articulation_points
        })

    return pd.DataFrame(analysis).sort_values(by="betweenness", ascending=False)


def get_high_degree_nodes(junction_csv: str, id_col: str, threshold: int = 500):
    df = pd.read_csv(junction_csv)
    counts = df[id_col].value_counts()
    high_degree_nodes = counts[counts > threshold].index.tolist()
    return set(high_degree_nodes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--blacklist", action="store_true")
    parser.add_argument("--binning", action="store_true")
    args = parser.parse_args()

    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]

    # --- Assign Components to Splits. Creates subsets for all splits for entities
    splits = split_dataset()

    # --- Pair Generation ---
    processed_entities: Dict[str, processedEntity] = {}

    for cfg_name, cfg in CONFIGS.items():
        print(f"Processing entity: {cfg.name}")

        # Extract IDs specific to this entity from the global splits
        def get_ids(comps: list[Dict[str, Set[str]]]):
            # for every component in components
            # return every node of component[current entity type]
            return {node for c in comps for node in c[cfg_name]}

        train_ids, valid_ids, test_ids = map(get_ids, splits)

        df_basics = pd.read_csv(cfg.path_basics)
        df_basics = df_basics.drop(columns=cfg.drop_list)
        # df_basics = df_basics.set_index(cfg.id_col)
        uf: UnionFind = entity_ufs[cfg.name]
        # map every entity to its root
        mapping = {entity: uf.find(entity) for entity in uf.parent.keys()}

        def prep_subset(ids):
            df = df_basics[df_basics[cfg.id_col].isin(ids)].copy()
            df['cluster_id'] = df[cfg.id_col].map(mapping)
            df['block_key'] = df.apply(cfg.block_key_func, axis=1)
            df = df.set_index(cfg.id_col)
            return df

        train_df, valid_df, test_df = map(
            prep_subset, [train_ids, valid_ids, test_ids])

        # Generate Pairs
        p_train, p_valid, p_test = map(generate_pairs_for_subset, [
                                       train_df, valid_df, test_df])

        for pairs in [p_train, p_valid, p_test]:
            random.shuffle(pairs)

        # Store for saving and inference later
        processed_entities[cfg.name] = processedEntity(
            uf, df_basics, {"train": p_train, "valid": p_valid, "test": p_test}, [], {}, {})

    for name, ids in [("Train", train_ids), ("Test", test_ids)]:
        roots = {entity_ufs[cfg_name].find(
            i) for i in ids if i in entity_ufs[cfg_name].parent}
        print(f"{name} unique roots: {len(roots)}")

    # --- Pairs for Inference ---
    # Create cartesian product for all related entries
    # this should be done in serialization module?
    for cfg_name, cfg in CONFIGS.items():
        main_test_pairs = processed_entities[cfg.name].pairs["test"]
        for rel in cfg.rels:
            rel_name = rel["rel_name"]
            print(f"Generating cp for: {cfg_name} {rel_name}")
            m_to_d_map = relation_maps[cfg_name+rel_name]

            # Create cartesian product pairs
            p_inference = propagate_dependency_pairs(
                main_test_pairs, m_to_d_map)

            # Label them
            labeled_inference = add_labels(
                p_inference,
                processed_entities[rel_name].uf,
                processed_entities[rel_name].df,
                CONFIGS[rel_name].id_col
            )

            try:
                processed_entities[rel_name].cp = processed_entities[rel_name].cp + \
                    labeled_inference
            except KeyError:
                print("Key Error!")
                processed_entities[rel_name].cp = labeled_inference

    # --- Pollution & Smaller Train Sets ---
    for cfg_name, cfg in CONFIGS.items():

        dfs_by_pollution: Dict = pollute(
            cfg.path_basics, cfg.id_col, cfg.drop_list)

        processed_entities[cfg.name].dfs_by_pollution = dfs_by_pollution

        pairs_full = processed_entities[cfg.name].pairs["train"]
        labels = [p[2] for p in pairs_full]

        train_log = get_log_subsets(pairs_full, labels, 5, 3, 6)

        processed_entities[cfg.name].train_log = train_log

    with open(f"pickles/{args.dataset}_processed_entities.pickle", 'wb') as f:
        pickle.dump(processed_entities, f, pickle.HIGHEST_PROTOCOL)

    with open(f"pickles/{args.dataset}_relation_maps.pickle", 'wb') as f:
        pickle.dump(relation_maps, f, pickle.HIGHEST_PROTOCOL)

    # --- Write Splits to Disk ---
    for cfg_name, cfg in CONFIGS.items():
        # Write splits for each pollution level
        for level, df in processed_entities[cfg.name].dfs_by_pollution.items():
            write_splits(cfg, CONFIGS, processed_entities,
                         relation_maps, level, args.binning)

    # --- Check requirements ---
    # No leakage
    # Relational Evidence remains
    validate_splits(splits, global_rel_map, entity_ufs)
    metadata_fp = f"{args.dataset}_metadata.json"
    generate_metadata(args, components, processed_entities, metadata_fp)
    print_overlap_table(processed_entities)

    copied = set()
    for cfg in CONFIGS.values():
        for relation in cfg.rels:
            # Copy junction table to new dir
            if not cfg.path_out_dir in copied:

                file_name = Path(relation['junction_table']).name

                shutil.copyfile(
                    relation["junction_table"], f"{cfg.path_out_dir}/{file_name}")
            copied.add(f"{cfg.path_out_dir}/{file_name}")


if __name__ == "__main__":
    main()
