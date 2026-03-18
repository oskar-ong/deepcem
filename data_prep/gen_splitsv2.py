from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import dataclass
import itertools
from pathlib import Path
import pickle
import random
from collections import defaultdict
from queue import Queue
from typing import Dict, List, Literal, Set, Tuple

import pandas as pd
import networkx as nx

from entityConfig import REGISTRY, EntityConfig

# --- Parameters ---
SPLIT_RATIOS   = (0.7, 0.1, 0.2)  # Train, Val, Test
NEG_RATIO      = 3                # Negatives per 1 Positive
RANDOM_SEED    = 0
BLOCK_LIMIT    = 10               # Max records per block to avoid N^2 growth
SplitMode = Literal["count", "nodes"]

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

def find_connected_components(rel_map: Dict[str, Set[str]]) -> list[Dict[str, Set[str]]]:
    used = set()
    components = []
    # for every entity in relation map
    for node in rel_map.keys():
        # disregard already seen entities
        if node in used: continue
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
            ent_comp = comp[u[1]] # second element of the tuple denotes entity type
            ent_comp.add(u[0]) # add node to its entity type set
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
) -> Tuple[list[Dict[str, Set[str]]],list[Dict[str, Set[str]]],list[Dict[str, Set[str]]]]:
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
        total_nodes = sum(len(c["all_nodes"]) for c in comps_list)
        # assign absolute ratios to splits
        targets = {"train": r_train * total_nodes, "val": r_val * total_nodes, "test": r_test * total_nodes}
        # sort components by number of nodes
        comps_sorted = sorted(comps_list, key=lambda kv: len(kv["all_nodes"]), reverse=True)
        # split: (components, current number of nodes)
        splits = {"train": ([], 0.0), "val": ([], 0.0), "test": ([], 0.0)}

        # iterate over all components
        for comp in comps_sorted:
            all_nodes = comp["all_nodes"]
            # current component size (count member nodes)
            size = float(len(all_nodes))
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
    name_lookup = df_tmp.set_index(id_col, drop=False).to_dict('index')
    
    for n1, n2 in pairs:
        if n1 in name_lookup and n2 in name_lookup:
            label = 1 if uf.find(n1) == uf.find(n2) else 0
            labeled_pairs.append((name_lookup[n1], name_lookup[n2], label))
    return labeled_pairs

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
            c_max = 0.0 # current max score for this dependency
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
        
        monge_elkan = ( 1/len(deps_left) ) * sum(scores) 
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
            final_score = "UNC" # uncertain

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
                #unique_clusters = {entity_ufs[cfg.name].find(n) for n in nodes}
                uf = entity_ufs[ent_type]
                unique_clusters = {
                    uf.find(n) if n in uf.parent else n 
                    for n in nodes
                }
                row[f"{ent_type}_clusters"] = len(unique_clusters)
                row[f"{ent_type}_dupe_count"] = len(nodes) - len(unique_clusters)
            
            analysis_data.append(row)

    return pd.DataFrame(analysis_data)

def analyze_graph_centrality(global_rel_map):
    # 1. Build the NetworkX graph
    G = nx.Graph()
    for node, neighbors in global_rel_map.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    print(f"Graph stats: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

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

def get_high_degree_nodes(junction_csv: str, id_col: str, threshold: int= 500):
    df = pd.read_csv(junction_csv)
    counts = df[id_col].value_counts()
    high_degree_nodes = counts[counts > threshold].index.tolist()
    return set(high_degree_nodes)

@dataclass
class processedEntity:
    uf: UnionFind
    df: pd.DataFrame
    pairs: Dict[str, List[Tuple[dict, dict, int]]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    args = parser.parse_args()

    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]

    global_rel_map: Dict[Tuple[str, str], Set[Tuple[str, str]]] = defaultdict(set)
    entity_ufs: Dict[str, UnionFind] = {}
    relation_maps = {} # Store these for inference later

    for cfg_name, cfg in CONFIGS.items():
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
                global_rel_map[(node, cfg_name)].add((root, cfg_name)) 
                # root -> duplicate
                global_rel_map[(root, cfg_name)].add((node, cfg_name))

        for rel_dict in cfg.rels:
            rel_name = rel_dict["rel_name"]
            rel_cfg = CONFIGS[rel_name] 
            blacklist = get_high_degree_nodes(rel_dict["junction_table"], rel_cfg.id_col, threshold = 10)
            print(blacklist)
            m_to_d = build_relation_map(rel_dict["junction_table"], cfg.id_col, rel_cfg.id_col, blacklist)
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
    df_centrality.to_csv("graph_bottlenecks.csv", index=False)
    df_stats = profile_components(components, CONFIGS, entity_ufs)
    with open ('pickles/stats.pickle', 'wb') as f:
        pickle.dump(df_stats, f, pickle.HIGHEST_PROTOCOL)

    # ==========================================
    # END ANALYSIS 
    # ==========================================

    splits = assign_components_to_splits(components)

    # pickling to evaluate created components and splits in different file
    with open ('pickles/components.pickle', 'wb') as f:
        pickle.dump(components, f, pickle.HIGHEST_PROTOCOL)
    with open ('pickles/splits.pickle', 'wb') as f:
        pickle.dump(splits, f, pickle.HIGHEST_PROTOCOL)

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
        uf: UnionFind = entity_ufs[cfg.name]
        # map every entity to its root
        mapping = {entity: uf.find(entity) for entity in uf.parent.keys()}

        def prep_subset(ids):
            df = df_basics[df_basics[cfg.id_col].isin(ids)].copy()
            df['cluster_id'] = df[cfg.id_col].map(mapping)
            df['block_key'] = df.apply(cfg.block_key_func, axis=1)
            return df

        train_df, valid_df, test_df = map(prep_subset, [train_ids, valid_ids, test_ids])

        # Generate Pairs
        p_train, p_valid, p_test = map(generate_pairs_for_subset, [train_df, valid_df, test_df])

        for pairs in [p_train, p_valid, p_test]:
            random.shuffle(pairs)

        # Store for saving and inference later
        processed_entities[cfg.name] = processedEntity(uf, df_basics, {"train": p_train, "valid": p_valid, "test": p_test})

    # ==========================================
    # SPLITS AND PAIRS ARE NOW LOCKED!
    # CONTINUE WITH INDIVIDUAL EXPERIMENT 
    # NOT REALLY BEST PERFORMANCE BUT SPLIT FOR READABILITY / UNDERSTANDING
    # ==========================================

    # ==========================================
    # MAIN EXPERIMENT: 
    # Cartesian Product for each entity to be matched 
    # ==========================================
    for cfg_name, cfg in CONFIGS.items():
        main_test_pairs = processed_entities[cfg.name].pairs["test"]
        for rel in cfg.rels:
            rel_name = rel["rel_name"]
            print(f"Generating inference for dependent: {rel_name}")
            m_to_d_map = relation_maps[cfg_name+rel_name]
            
            # Create cartesian product pairs
            p_inference = propagate_dependency_pairs(main_test_pairs, m_to_d_map, cfg.id_col)
            
            # Label them
            labeled_inference = add_labels(
                p_inference, 
                processed_entities[rel_name].uf, 
                processed_entities[rel_name].df, 
                CONFIGS[rel_name].id_col
            )

            try:
                processed_entities[rel_name].pairs["cp"] = processed_entities[rel_name].pairs["cp"] +labeled_inference
            except KeyError:
                processed_entities[rel_name].pairs["cp"] = labeled_inference

    # ==========================================
    # Save to Disk & Ditto Serialization
    # ==========================================
    DROPOUT_PROB = 0.15

    for cfg_name, cfg in CONFIGS.items():
        pairs_dict = processed_entities[cfg.name].pairs

        def serialize(pairs: List[Tuple[dict, dict, int]]) -> List[str]:
            lines = []
            for pair in pairs:
                left = pair[0]
                right = pair[1]
                label = pair[2]

                l_part: str = ""
                r_part: str = ""

                # Helper to format one side into COL VAL strings
                def fmt(pair_part: Dict, drop_list):
                    return " ".join([f"COL {k} VAL {v}" for k, v in pair_part.items() if pd.notna(v) and k not in drop_list])

                l_part = fmt(left, cfg.drop_list)
                r_part = fmt(right, cfg.drop_list)

                line = f"{l_part}\t{r_part}\t{label}"
                lines.append(line)
            return lines
        
        for split, pairs in pairs_dict.items():

            if split in ["train", "valid", "test"]:
                # ========================================================================================================================================================================
                # BASELINE A: 
                # ========================================================================================================================================================================
                pairs_baselineA = copy.deepcopy(pairs)
                lines = serialize(pairs_baselineA)
                baseA_dir = f"{cfg.path_out_dir}baseA/"
                Path(baseA_dir).mkdir(parents=True, exist_ok=True)
                with open(f"{baseA_dir}{split}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                
                # ========================================================================================================================================================================
                # BASELINE B: 
                # ========================================================================================================================================================================

                # TODO: Add Relation Columns to all entries, even the ones with null values

                pairs_baselineB = copy.deepcopy(pairs)
                
                # get all unique ids in pairs
                ids = set([d[cfg.id_col] for d1, d2, _ in pairs for d in (d1, d2)])
                flat: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(dict)) # maincfg -> relation -> relation_attributes -> List of attribute values

                # preload all related dfs and set id as index
                indexed_dfs = {
                    rel["rel_name"]: processed_entities[rel["rel_name"]].df.set_index(CONFIGS[rel["rel_name"]].id_col)
                    for rel in cfg.rels
                }

                for m_id in ids:
                    for rel in cfg.rels:
                        rel_name = rel["rel_name"]
                        rel_map = relation_maps[cfg_name+rel_name]
                        related_entries = rel_map.get(m_id, [])

                        df = indexed_dfs[rel_name]
                        relation_attributes = defaultdict(list)

                        for entry in related_entries:
                            try: 
                                row = df.loc[entry]
                            
                                if isinstance(row, pd.DataFrame):
                                    raise LookupError(f"More than 1 entry for ID {entry}")
                                
                                for col_name, value in row.items():
                                    relation_attributes[col_name].append(value)

                            except KeyError:
                                continue

                            for col_name, value in row.items():
                                relation_attributes[col_name].append(value)

                        flat[m_id][rel_name] = dict(relation_attributes)
                for left, right, label in pairs_baselineB:
                    l_id = left.get(cfg.id_col)
                    extra_data = flat.get(l_id, {})

                    # The transformation
                    flattened = {
                        f"{rel}_{attr}": " ".join(str(v) for v in values)
                        for rel, attributes in extra_data.items() 
                        for attr, values in attributes.items()
                    }

                    left.update(flattened)

                    r_id = right.get(cfg.id_col)
                    extra_data = flat.get(r_id, {})
                    flattened = {
                        f"{rel}_{attr}": " ".join(str(v) for v in values)
                        for rel, attributes in extra_data.items() 
                        for attr, values in attributes.items()
                    }
                    right.update(flattened)


                lines = serialize(pairs_baselineB)
                baseB_dir = f"{cfg.path_out_dir}baseB/"
                Path(baseB_dir).mkdir(parents=True, exist_ok=True)
                with open(f"{baseB_dir}{split}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")

            # ========================================================================================================================================================================
            # SCORES EMPTY: 
            # ========================================================================================================================================================================
            pairs_empty_scores = copy.deepcopy(pairs)

            for rel in cfg.rels:
                rel_name = rel["rel_name"]
                for left, right, label in pairs_empty_scores:
                    left[f"{rel_name}_score"] = ""
                    right[f"{rel_name}_score"] = ""

            lines = serialize(pairs_empty_scores)
            empty_scores_dir = f"{cfg.path_out_dir}emptyScores/"
            Path(empty_scores_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{empty_scores_dir}{split}.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")

            # ========================================================================================================================================================================
            # SCORES INJECTED: 
            # ========================================================================================================================================================================
            pairs_injected_scores = copy.deepcopy(pairs_empty_scores)

            for rel in cfg.rels:
                rel_name = rel["rel_name"]
                for left, right, label in pairs_injected_scores:
                    score = calculate_relationship_scores(
                        left[cfg.id_col], 
                        right[cfg.id_col], 
                        relation_maps[cfg_name+rel_name], 
                        processed_entities[rel_name].uf,
                        DROPOUT_PROB,
                        False)
                    left[f"{rel_name}_score"] = score
                    right[f"{rel_name}_score"] = score

            lines = serialize(pairs_injected_scores)
            injected_scores_dir = f"{cfg.path_out_dir}injectedScores/"
            Path(injected_scores_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{injected_scores_dir}{split}.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")

            # ========================================================================================================================================================================
            # DONE!!!
            # ========================================================================================================================================================================

if __name__ == "__main__":
    main()