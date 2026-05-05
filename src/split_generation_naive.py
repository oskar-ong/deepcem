from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import pickle
import random

import shutil
from typing import Dict

import pandas as pd

from data_structures import UnionFind, processedEntity
from serializer import write_splits
from entity_config import REGISTRY, EntityConfig
from pollution import pollute
from split_generation import add_labels, build_unionfind_with_singletons, propagate_dependency_pairs


def generate_pos_pairs(df):
    pos_pairs = []
    clusters = df.groupby('cluster_id').groups
    for cluster_id, indices in clusters.items():
        if len(indices) < 2:
            continue
        for p1, p2 in itertools.combinations(indices, 2):
            pos_pairs.append((p1, p2, 1))
    return pos_pairs


def generate_neg_pairs(df, n_neg):
    hard_negs = []
    blocks = df.groupby('block_key').groups

    block_keys = list(blocks.keys())
    random.shuffle(block_keys)  # Shuffle blocks for variety

    for key in block_keys:
        indices = list(blocks[key])
        if len(indices) < 2:
            continue

        # Try to find hard negs in this block without exhaustive O(n^2)
        random.shuffle(indices)
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                id1, id2 = indices[i], indices[j]
                if df.at[id1, 'cluster_id'] != df.at[id2, 'cluster_id']:
                    hard_negs.append((id1, id2, 0))
                if len(hard_negs) >= n_neg:
                    return hard_negs

    easy_negs = []
    remaining = n_neg - len(hard_negs)
    ids = df.index.tolist()

    while len(easy_negs) < remaining:
        id1, id2 = random.sample(ids, 2)
        if df.at[id1, 'cluster_id'] != df.at[id2, 'cluster_id']:
            easy_negs.append((id1, id2, 0))

    return hard_negs + easy_negs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("pos_pairs", type=int)
    parser.add_argument("ratio", type=int)
    args = parser.parse_args()

    CONFIGS: Dict[str, EntityConfig] = REGISTRY[args.dataset]
    n_pos = args.pos_pairs
    ratio = args.ratio
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
        pos_pairs = generate_pos_pairs(df)
        total_pos_count = len(pos_pairs)

        if total_pos_count < n_pos:
            raise ValueError(
                f"Not enough positive pairs available! Needed: {n_pos}, aviable: {total_pos_count}")

        total_neg_count = total_pos_count * args.ratio

        neg_pairs = generate_neg_pairs(df, total_neg_count)

        random.shuffle(pos_pairs)
        random.shuffle(neg_pairs)

        train_pos = pos_pairs[:n_pos]
        print(len(train_pos))
        train_neg = neg_pairs[:n_pos * ratio]
        train = (train_pos + train_neg)
        random.shuffle(train)
        print(len(train))

        remaining_pos = pos_pairs[n_pos:]
        remaining_neg = neg_pairs[n_pos * ratio:]

        remaining_pairs = remaining_pos + remaining_neg

        split_idx = int(len(remaining_pairs) * 0.3)
        val = remaining_pairs[:split_idx]  # first third
        random.shuffle(val)
        test = remaining_pairs[split_idx:]  # other two thirds
        random.shuffle(test)

        # Store for saving and inference later
        processed_entities[cfg.name] = processedEntity(
            uf, df_basics, {"train": train, "valid": val, "test": test}, [], {}, {})

    return None
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

        # pairs_full = processed_entities[cfg.name].pairs["train"]
        # labels = [p[2] for p in pairs_full]

        # train_log = get_log_subsets(pairs_full, labels, 5, 3, 6)

        # processed_entities[cfg.name].train_log = train_log

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
