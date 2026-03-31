from __future__ import annotations
import json
import argparse
import csv
from dataclasses import dataclass
import itertools
import pickle
import random
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Literal, Set, Tuple

import pandas as pd
import networkx as nx

from data_structures import UnionFind, processedEntity
from serializer import write_splits
from entityConfig import REGISTRY, EntityConfig
from pollution import pollute

# --- Parameters ---
SPLIT_RATIOS = (0.7, 0.1, 0.2)  # Train, Val, Test
NEG_RATIO = 3                # Negatives per 1 Positive
RANDOM_SEED = 0
BLOCK_LIMIT = 10               # Max records per block to avoid N^2 growth
SplitMode = Literal["count", "nodes"]


def print_overlap_table(CONFIGS: Dict[str, EntityConfig], processed_entities: Dict[str, processedEntity]):
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

        # Get related actors for both movies
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
    # for n1, n2 in pairs:
    #     if n1 in name_lookup and n2 in name_lookup:
    #         label = 1 if uf.find(n1) == uf.find(n2) else 0
    #         labeled_pairs.append((name_lookup[n1], name_lookup[n2], label))
    #


def calculate_relationship_scores(left_id, right_id, entity_to_deps, dep_uf, dropout_prob, is_bin=False):
    # Monge elkan
    final_score = 0.5

    if is_bin == True:
        final_score = "UNC"

    # Get dependent entities
    deps_left = entity_to_deps.get(left_id, set())
    deps_right = entity_to_deps.get(right_id, set())

    if len(deps_right) < len(deps_left):
        tmp = deps_right
        deps_right = deps_left
        deps_left = tmp

    scores = []

    if deps_left and deps_right:
        for d_left in deps_left:
            c_max = 0.0  # current max score for this dependency
            for d_right in deps_right:
                if d_left in dep_uf.parent and d_right in dep_uf.parent:
                    if dep_uf.find(d_left) == dep_uf.find(d_right):
                        score = 1
                    else:
                        score = 0
                else:
                    score = 0
                if score > c_max:
                    c_max = score
            scores.append(c_max)

        monge_elkan = (1/len(deps_left)) * sum(scores)
    else:
        monge_elkan = 0.5

    # Signal Dropout logic
    # final_score = 0.5 if random.random() < dropout_prob else max_pool_score
    final_score = 0.5 if random.random() < dropout_prob else round(monge_elkan, 2)

    # BINNING
    if is_bin == True:
        if final_score >= 0.85:
            final_score = "HIGH"
        if final_score <= 0.15:
            final_score = "LOW"
        if 0.15 < final_score < 0.85:
            final_score = "UNC"  # uncertain

    return final_score


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
    # 1. Build the NetworkX graph
    G = nx.Graph()
    for node, neighbors in global_rel_map.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    print(
        f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # 2. Find Articulation Points (Cut Vertices)
    # These are nodes that, if removed, increase the number of connected components.
    articulation_points = list(nx.articulation_points(G))

    # 3. Calculate Betweenness Centrality
    # This measures how often a node appears on the shortest path between any two other nodes.
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
    args = parser.parse_args()

    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]

    global_rel_map: Dict[Tuple[str, str],
                         Set[Tuple[str, str]]] = defaultdict(set)
    entity_ufs: Dict[str, UnionFind] = {}
    relation_maps = {}  # Store these for inference later
    splitting_blacklist = set()

    # Dynamic Blacklist
    do_blacklist = False
    if do_blacklist == True:
        for cfg_name, cfg in CONFIGS.items():
            for rel_dict in cfg.rels:
                # We prune the top 5% of most common relations to break the Giant Component
                hubs = get_dynamic_blacklist(
                    rel_dict["junction_table"], CONFIGS[rel_dict["rel_name"]].id_col, percentile=0.95)
                splitting_blacklist.update(hubs)

    for cfg_name, cfg in CONFIGS.items():
        # create a union find for each entity type based on duplicate csv (transitive closure)
        uf = build_unionfind_with_singletons(
            cfg.path_basics, cfg.path_dups, cfg.id_col)
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
                global_rel_map[(node, cfg_name)].add((root, cfg_name))
                # root -> duplicate
                global_rel_map[(root, cfg_name)].add((node, cfg_name))

        for rel_dict in cfg.rels:
            rel_name = rel_dict["rel_name"]
            rel_cfg = CONFIGS[rel_name]

            # blacklist = get_high_degree_nodes(rel_dict["junction_table"], rel_cfg.id_col, threshold = 10)
            # print(blacklist)
            m_to_d = build_relation_map(
                rel_dict["junction_table"], cfg.id_col, rel_cfg.id_col, splitting_blacklist)
            relation_maps[cfg_name+rel_name] = m_to_d

            for m_id, d_ids in m_to_d.items():
                for d_id in d_ids:
                    global_rel_map[(m_id, cfg_name)].add((d_id, rel_name))
                    global_rel_map[(d_id, rel_name)].add((m_id, cfg_name))

    components = find_connected_components(global_rel_map)

    # ==========================================
    # ANALYSIS
    # ==========================================

    # Identify the largest component
    largest_comp_size = max(len(c["all_nodes"]) for c in components)
    print(f"Largest component size: {largest_comp_size}")

    # Analyze the bottlenecks
    df_centrality = analyze_graph_centrality(global_rel_map)
    print("Top 10 nodes responsible for connectivity:")
    print(df_centrality.head(10))

    # Export for inspection
    df_centrality.to_csv(f"{args.dataset}_graph_bottlenecks.csv", index=False)
    df_stats = profile_components(components, CONFIGS, entity_ufs)
    with open(f"pickles/{args.dataset}_stats.pickle", 'wb') as f:
        pickle.dump(df_stats, f, pickle.HIGHEST_PROTOCOL)

    # ==========================================
    # END ANALYSIS
    # ==========================================

    splits = assign_components_to_splits(components)

    # pickling to evaluate created components and splits in different file
    with open(f"pickles/{args.dataset}_components.pickle", 'wb') as f:
        pickle.dump(components, f, pickle.HIGHEST_PROTOCOL)
    with open(f"pickles/{args.dataset}_splits.pickle", 'wb') as f:
        pickle.dump(splits, f, pickle.HIGHEST_PROTOCOL)
    with open(f"pickles/{args.dataset}_relmap.pickle", 'wb') as f:
        pickle.dump(global_rel_map, f, pickle.HIGHEST_PROTOCOL)

    # ==========================================
    # Generic Prep & Pair Generation
    # ==========================================
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
            uf, df_basics, {"train": p_train, "valid": p_valid, "test": p_test}, [], {})

    for name, ids in [("Train", train_ids), ("Test", test_ids)]:
        roots = {entity_ufs[cfg_name].find(
            i) for i in ids if i in entity_ufs[cfg_name].parent}
        print(f"{name} unique roots: {len(roots)}")
    # ==========================================
    # SPLITS AND PAIRS ARE NOW LOCKED!
    # CONTINUE WITH INDIVIDUAL EXPERIMENT
    # NOT REALLY BEST PERFORMANCE BUT SPLIT FOR READABILITY / UNDERSTANDING
    # ==========================================

    # --- Pairs for Inference ---
    # Create cartesian product for all related entries
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

    # --- Pollution ---
    for cfg_name, cfg in CONFIGS.items():

        dfs_by_pollution: Dict = pollute(
            cfg.path_basics, cfg.id_col, cfg.drop_list)

        processed_entities[cfg.name].dfs_by_pollution = dfs_by_pollution

    # --- Write Splits to Disk ---
    for cfg_name, cfg in CONFIGS.items():
        # Write splits for each pollution level
        for level, df in processed_entities[cfg.name].dfs_by_pollution.items():
            write_splits(cfg, CONFIGS, processed_entities,
                         relation_maps, level)

    # ========================================================================================================================================================================
    # Check requirements:
    # No leakage
    # Relational Evidence remains
    # ========================================================================================================================================================================
    validate_splits(splits, global_rel_map, entity_ufs)
    metadata_fp = f"{args.dataset}_metadata.json"
    generate_metadata(args, components, processed_entities, metadata_fp)
    print_overlap_table(CONFIGS, processed_entities)


if __name__ == "__main__":
    main()
