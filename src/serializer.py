from collections import defaultdict
import copy
import json
from pathlib import Path
import random
from typing import Dict, List, Tuple

import pandas as pd

from entity_config import EntityConfig
from data_structures import processedEntity
DROPOUT_PROB = 0.2


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
    final_score = 0.5 if random.random() < dropout_prob else round(monge_elkan, 2)

    # BINNING
    if is_bin == True:
        if final_score >= 0.67:
            final_score = "high"
        elif final_score <= 0.33:
            final_score = "low"
        elif 0.33 < final_score < 0.67:
            final_score = "uncertain"

    return final_score


def serialize(pairs: List[Tuple[dict, dict, int]], cfg: EntityConfig) -> List[str]:
    lines = []
    amt_pos = 0
    amt_neg = 0
    for pair in pairs:
        left = pair[0]
        right = pair[1]
        label = pair[2]
        if label == 1:
            amt_pos += 1
        if label == 0:
            amt_neg += 1

        l_part: str = ""
        r_part: str = ""

        # Helper to format one side into COL VAL strings
        def fmt(pair_part: Dict, drop_list):
            return " ".join([f"COL {k} VAL {v}" for k, v in pair_part.items() if pd.notna(v) and k not in drop_list])

        l_part = fmt(left, cfg.drop_list)
        r_part = fmt(right, cfg.drop_list)

        line = f"{l_part}\t{r_part}\t{label}"
        lines.append(line)
    return lines, amt_pos, amt_neg


def write_splits(cfg: EntityConfig, CONFIGS, processed_entities: Dict[str, processedEntity], relation_maps, level, is_bin=False):

    attributes = processed_entities[cfg.name].dfs_by_pollution[level].to_dict(
        "index")

    # Inject the index as an attribute
    for idx, record in attributes.items():
        record['id'] = idx

    for split, pairs_ids in processed_entities[cfg.name].pairs.items():
        pairs: List[Tuple[Dict, Dict, int]] = []
        # load attribtues for ids
        for id1, id2, label in pairs_ids:
            record1 = attributes.get(id1)
            record2 = attributes.get(id2)

            if record1 and record2:
                pairs.append((record1, record2, label))

        if split in ["train", "valid", "test"]:
            # --- Baseline A: attributes only ---

            pairs_baselineA = copy.deepcopy(pairs)
            lines, amt_pos, amt_neg = serialize(pairs_baselineA, cfg)

            exp_name = "baseA"
            baseA_dir = Path(
                f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")

            Path(baseA_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{baseA_dir}/{split}.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
            print(
                f"Wrote {len(lines)} lines for BaselineA {cfg.name} {split}. Pos: {amt_pos}, Neg: {amt_neg}")

            if split == "train":
                for log_power, pair_ids_log in processed_entities[cfg.name].train_log.items():
                    pairs_log: List[Tuple[Dict, Dict, int]] = []
                    for id1, id2, label in pair_ids_log:
                        record1 = attributes.get(id1)
                        record2 = attributes.get(id2)

                        if record1 and record2:
                            pairs_log.append((record1, record2, label))

                    lines, amt_pos, amt_neg = serialize(pairs_log, cfg)
                    with open(f"{baseA_dir}/{split}_{log_power}.txt", 'w', encoding='utf-8') as f:
                        f.write("\n".join(lines) + "\n")
                    print(
                        f"Wrote {len(lines)} lines for BaselineA {cfg.name} {split}. Pos: {amt_pos}, Neg: {amt_neg}")

            # --- Baseline B: Flattened Schema ---
            # TODO: Add Relation Columns to all entries, even the ones with null values
            pairs_baselineB = copy.deepcopy(pairs)
            # get all unique ids in pairs
            ids = set([d['id']
                      for d1, d2, _ in pairs for d in (d1, d2)])
            flat: Dict[str, Dict[str, Dict[str, List[str]]]] = defaultdict(lambda: defaultdict(
                dict))  # maincfg -> relation -> relation_attributes -> List of attribute values

            # preload all related dfs and set id as index
            indexed_dfs = {
                rel["rel_name"]: processed_entities[rel["rel_name"]
                                                    ].df.set_index(CONFIGS[rel["rel_name"]].id_col)
                for rel in cfg.rels
            }

            for m_id in ids:
                for rel in cfg.rels:
                    rel_name = rel["rel_name"]
                    rel_map = relation_maps[cfg.name+rel_name]
                    related_entries = rel_map.get(m_id, [])

                    df = indexed_dfs[rel_name]
                    relation_attributes = defaultdict(list)

                    for entry in related_entries:
                        try:
                            row = df.loc[entry]

                            if isinstance(row, pd.DataFrame):
                                raise LookupError(
                                    f"More than 1 entry for ID {entry}")

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

            lines, amt_pos, amt_neg = serialize(pairs_baselineB, cfg)
            exp_name = "baseB"
            baseB_dir = Path(
                f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")
            Path(baseB_dir).mkdir(parents=True, exist_ok=True)
            with open(f"{baseB_dir}/{split}.txt", 'w', encoding='utf-8') as f:
                f.write("\n".join(lines) + "\n")
            print(
                f"Wrote {len(lines)} lines for BaselineB {cfg.name} {split}. Pos: {amt_pos}, Neg: {amt_neg}")

        # --- Scores Empty ---
        pairs_empty_scores = copy.deepcopy(pairs)

        sim_keys = {
            f"{cfg.name}_{rel['rel_name']}_similarity": "" for rel in cfg.rels}

        processed_empty = []
        for left, right, label in pairs:  # Use the original 'pairs'
            # 2. Create fresh, independent copies for THIS pair only
            new_left = copy.deepcopy(left)
            new_right = copy.deepcopy(right)

            # 3. Build a NEW dict for left (scores first, then original data)
            # This is non-destructive and won't affect other pairs
            prefix_left = {**sim_keys, **new_left}

            # 4. Ensure right is clean of any similarity keys
            for k in list(new_right.keys()):
                if "similarity" in k:
                    del new_right[k]

            processed_empty.append((prefix_left, new_right, label))

        lines, amt_pos, amt_neg = serialize(processed_empty, cfg)
        exp_name = "emptyScores"
        empty_scores_dir = Path(
            f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")
        Path(empty_scores_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{empty_scores_dir}/{split}.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(
            f"Wrote {len(lines)} lines for empty scores {cfg.name} {split}. Pos: {amt_pos}, Neg: {amt_neg}")

        if split == "train":
            for log_power, pair_ids_log in processed_entities[cfg.name].train_log.items():
                pairs_log: List[Tuple[Dict, Dict, int]] = []
                for id1, id2, label in pair_ids_log:
                    record1 = attributes.get(id1)
                    record2 = attributes.get(id2)

                    if record1 and record2:
                        pairs_log.append((record1, record2, label))

                processed_train = []
                for left, right, label in pairs_log:
                    new_left = copy.deepcopy(left)
                    new_right = copy.deepcopy(right)

                    # 1. Calculate scores
                    injected_scores = {}
                    for rel in cfg.rels:
                        rel_name = rel["rel_name"]
                        injected_scores[f"{cfg.name}_{rel_name}_similarity"] = ""

                    # 2. Build the prefixed left dictionary
                    prefix_left = {**injected_scores, **new_left}

                    # 3. Clean the right dictionary
                    for k in list(new_right.keys()):
                        if "similarity" in k:
                            del new_right[k]

                    processed_train.append((prefix_left, new_right, label))

                lines, amt_pos, amt_neg = serialize(processed_train, cfg)
                with open(f"{empty_scores_dir}/{split}_{log_power}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                print(
                    f"Wrote {len(lines)} lines for empty scores {cfg.name} {split}_{log_power}. Pos: {amt_pos}, Neg: {amt_neg}")

        if split == "test":
            total_pairs = 0
            amt_pos = 0
            amt_neg = 0
            exp_name = "inference"
            inference_dir = Path(
                f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")
            Path(inference_dir).mkdir(parents=True, exist_ok=True)

            with open(f"{inference_dir}/test_labeled.jsonl", 'w', encoding='utf-8') as f_lab, \
                    open(f"{inference_dir}/test_unlabeled.jsonl", 'w', encoding='utf-8') as f_unlab:

                sim_keys = {
                    f"{cfg.name}_{rel['rel_name']}_similarity": "" for rel in cfg.rels}

                for left, right, label in pairs_empty_scores:

                    new_left = copy.deepcopy(left)
                    new_right = copy.deepcopy(right)

                    # 3. Build a NEW dict for left (scores first, then original data)
                    # This is non-destructive and won't affect other pairs
                    prefix_left = {**sim_keys, **new_left}

                    # 4. Ensure right is clean of any similarity keys
                    for k in list(new_right.keys()):
                        if "similarity" in k:
                            del new_right[k]

                    tmp_left = copy.deepcopy(prefix_left)
                    tmp_right = copy.deepcopy(new_right)

                    prefix_left.clear()
                    new_right.clear()
                    for k, v in tmp_left.items():
                        if k not in cfg.drop_list:
                            prefix_left[k] = v
                    for k, v in tmp_right.items():
                        if k not in cfg.drop_list:
                            new_right[k] = v

                    if label == 1 or str(label).lower() == 'true':
                        amt_pos += 1
                    else:
                        amt_neg += 1

                    # 4. Create the labeled and unlabeled objects
                    pair_list = [prefix_left, new_right]
                    f_unlab.write(json.dumps(
                        pair_list, ensure_ascii=False) + "\n")

                    # 5. Write each as a single line in their respective files
                    labeled_list = [prefix_left, new_right, label]
                    f_lab.write(json.dumps(
                        labeled_list, ensure_ascii=False) + "\n")

                    total_pairs += 1

                print(
                    f"Successfully wrote {total_pairs} lines to JSONL files. Pos: {amt_pos}, Neg: {amt_neg}")

        # --- Scores Injected ---
        processed_injected = []
        for left, right, label in pairs:  # Use original 'pairs'
            new_left = copy.deepcopy(left)
            new_right = copy.deepcopy(right)

            # 1. Calculate scores
            injected_scores = {}
            for rel in cfg.rels:
                rel_name = rel["rel_name"]
                score = calculate_relationship_scores(
                    new_left['id'], new_right['id'],
                    relation_maps[cfg.name+rel_name],
                    processed_entities[rel_name].uf,
                    DROPOUT_PROB, is_bin)
                injected_scores[f"{cfg.name}_{rel_name}_similarity"] = score

            # 2. Build the prefixed left dictionary
            prefix_left = {**injected_scores, **new_left}

            # 3. Clean the right dictionary
            for k in list(new_right.keys()):
                if "similarity" in k:
                    del new_right[k]

            processed_injected.append((prefix_left, new_right, label))

        lines, amt_pos, amt_neg = serialize(processed_injected, cfg)
        exp_name = "injectedScores"
        injected_scores_dir = Path(
            f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")
        Path(injected_scores_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{injected_scores_dir}/{split}.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(lines) + "\n")
        print(
            f"Wrote {len(lines)} lines for injected scores {cfg.name} {split}. Pos: {amt_pos}, Neg: {amt_neg}")

        if split == "train":
            for log_power, pair_ids_log in processed_entities[cfg.name].train_log.items():
                pairs_log: List[Tuple[Dict, Dict, int]] = []
                for id1, id2, label in pair_ids_log:
                    record1 = attributes.get(id1)
                    record2 = attributes.get(id2)

                    if record1 and record2:
                        pairs_log.append((record1, record2, label))

                processed_train = []
                for left, right, label in pairs_log:
                    new_left = copy.deepcopy(left)
                    new_right = copy.deepcopy(right)

                    # 1. Calculate scores
                    injected_scores = {}
                    for rel in cfg.rels:
                        rel_name = rel["rel_name"]
                        score = calculate_relationship_scores(
                            new_left['id'], new_right['id'],
                            relation_maps[cfg.name+rel_name],
                            processed_entities[rel_name].uf,
                            DROPOUT_PROB, is_bin)
                        injected_scores[f"{cfg.name}_{rel_name}_similarity"] = score

                    # 2. Build the prefixed left dictionary
                    prefix_left = {**injected_scores, **new_left}

                    # 3. Clean the right dictionary
                    for k in list(new_right.keys()):
                        if "similarity" in k:
                            del new_right[k]

                    processed_train.append((prefix_left, new_right, label))

                lines, amt_pos, amt_neg = serialize(processed_train, cfg)
                with open(f"{injected_scores_dir}/{split}_{log_power}.txt", 'w', encoding='utf-8') as f:
                    f.write("\n".join(lines) + "\n")
                print(
                    f"Wrote {len(lines)} lines for injected scores {cfg.name} {split}_{log_power}. Pos: {amt_pos}, Neg: {amt_neg}")

    # --- Prediction Jsons ---
    pairs_cp: List[str, str, int] = copy.deepcopy(
        processed_entities[cfg.name].cp)

    pairs: List[Tuple[Dict, Dict, int]] = []
    # load attribtues for ids
    for id1, id2, label in pairs_cp:
        record1 = attributes.get(id1)
        record2 = attributes.get(id2)

        if record1 and record2:
            pairs.append((record1, record2, label))
    # Define directory and ensure it exists
    exp_name = "inference"
    cp_dir = Path(
        f"{cfg.path_out_dir}/{level}/{exp_name}/{cfg.name}")
    Path(cp_dir).mkdir(parents=True, exist_ok=True)
    amt_pos = 0
    amt_neg = 0
    total_pairs = 0

    with open(f"{cp_dir}/cp_labeled.jsonl", 'w', encoding='utf-8') as f_lab, \
            open(f"{cp_dir}/cp_unlabeled.jsonl", 'w', encoding='utf-8') as f_unlab:

        sim_keys = {
            f"{cfg.name}_{rel['rel_name']}_similarity": "" for rel in cfg.rels}

        for left, right, label in pairs:

            new_left = copy.deepcopy(left)
            new_right = copy.deepcopy(right)

            # 3. Build a NEW dict for left (scores first, then original data)
            # This is non-destructive and won't affect other pairs
            prefix_left = {**sim_keys, **new_left}

            # 4. Ensure right is clean of any similarity keys
            for k in list(new_right.keys()):
                if "similarity" in k:
                    del new_right[k]

            tmp_left = copy.deepcopy(prefix_left)
            tmp_right = copy.deepcopy(new_right)

            prefix_left.clear()
            new_right.clear()
            for k, v in tmp_left.items():
                if k not in cfg.drop_list:
                    prefix_left[k] = v
            for k, v in tmp_right.items():
                if k not in cfg.drop_list:
                    new_right[k] = v

            if label == 1 or str(label).lower() == 'true':
                amt_pos += 1
            else:
                amt_neg += 1

            # 4. Create the labeled and unlabeled objects
            pair_list = [prefix_left, new_right]
            f_unlab.write(json.dumps(
                pair_list, ensure_ascii=False) + "\n")

            # 5. Write each as a single line in their respective files
            labeled_list = [prefix_left, new_right, label]
            f_lab.write(json.dumps(
                labeled_list, ensure_ascii=False) + "\n")

            total_pairs += 1

    print(
        f"Successfully wrote {total_pairs} lines to JSONL files. Pos: {amt_pos}, Neg: {amt_neg}")
