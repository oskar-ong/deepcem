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


def generate_pos_pairs(df, n_required):
    """Samples n_required pairs that share the same cluster_id."""
    pos_pool = []
    # Group by cluster_id to find all actual matches
    clusters = df.groupby('cluster_id').groups

    for cluster_id, indices in clusters.items():
        if len(indices) < 2:
            continue
        # Generate all combinations in this cluster
        for p1, p2 in itertools.combinations(indices, 2):
            pos_pool.append((p1, p2, 1))

    random.shuffle(pos_pool)
    return pos_pool[:n_required]


def generate_hard_neg(df, n_required):
    """Samples pairs with the same block_key but different cluster_id."""
    hard_neg_pool = []
    blocks = df.groupby('block_key').groups

    for block_key, indices in blocks.items():
        if len(indices) < 2:
            continue

        # We sample pairs within the block to avoid O(N^2) if a block is huge
        # Try to find hard negatives in this block
        block_list = list(indices)
        for _ in range(len(block_list) * 2):  # heuristic attempt limit
            id1, id2 = random.sample(block_list, 2)
            if df.at[id1, 'cluster_id'] != df.at[id2, 'cluster_id']:
                hard_neg_pool.append((id1, id2, 0))

            if len(hard_neg_pool) >= n_required:
                return hard_neg_pool

    random.shuffle(hard_neg_pool)
    return hard_neg_pool[:n_required]


def generate_easy_neg(df, n_required):
    """Samples pairs at random that have different cluster_ids."""
    easy_neg_pool = []
    ids = df.index.tolist()

    while len(easy_neg_pool) < n_required:
        id1, id2 = random.sample(ids, 2)
        if df.at[id1, 'cluster_id'] != df.at[id2, 'cluster_id']:
            easy_neg_pool.append((id1, id2, 0))

    return easy_neg_pool


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("pos_pairs", type=int)
    parser.add_argument("neg_pairs", type=int)
    args = parser.parse_args()

    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]
    pos_pairs = args.pos_pairs
    neg_pairs = args.neg_pairs
    entity_ufs: Dict[str, UnionFind] = {}

    # --- Pair Generation ---
    processed_entities: Dict[str, processedEntity] = {}

    for cfg_name, cfg in CONFIGS.items():
        uf = build_unionfind_with_singletons(
            cfg.path_basics, cfg.path_dups, cfg.id_col)
        entity_ufs[cfg.name] = uf

        df_basics = pd.read_csv(cfg.path_basics)
        df_basics = df_basics.drop(columns=cfg.drop_list)
        uf: UnionFind = entity_ufs[cfg.name]
        # map every entity to its root
        mapping = {entity: uf.find(entity) for entity in uf.parent.keys()}

        def prep_df(df_basics):
            df = df_basics.copy()
            df['cluster_id'] = df[cfg.id_col].map(mapping)
            df['block_key'] = df.apply(cfg.block_key_func, axis=1)
            df = df.set_index(cfg.id_col)
            return df

        df = prep_df(df_basics)

        # positive sampling
        pos_pairs = generate_pos_pairs()

        # hard negative sampling
        hard_neg_pairs = generate_hard_neg()

        # easy negative sampling
        easy_neg_pairs = generate_easy_neg()

        # assign to train, valid, test split
        train, temp = train_test_split()
        val, test = train_test_split()

        for pairs in [train, val, test]:
            random.shuffle(pairs)

        # Store for saving and inference later
        processed_entities[cfg.name] = processedEntity(
            uf, df_basics, {"train": train, "valid": val, "test": test}, [], {}, {})

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
