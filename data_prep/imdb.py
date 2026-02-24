from __future__ import annotations
from collections import defaultdict
import csv
from queue import Queue
import random
from typing import Dict, List, Literal, Sequence, Set, Tuple


SplitMode = Literal["count", "nodes"]  # extendable

def build_relation_map(csv_fp: str, column1: str, column2: str) -> Dict[str, Set[str]]:
    relation_map: Dict[str,Set[str]] = defaultdict(set)

    with open(csv_fp, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader: 
            c1 = row[column1]
            c2 = row[column2]
            if c1 and c2: 
                relation_map[c1].add(c2)

    return dict(relation_map)

def find_connected_components(rel_map: Dict[str,Set[str]]) -> list[Set[str]]:

    used = set()

    components = []

    for node in rel_map.keys():
        if node in used:
            continue
        comp = set()
        queue = Queue()
        queue.put(node)
        used.add(node)
        while not queue.empty():
            u = queue.get()
            comp.add(u)
            for v in rel_map[u]:
                if v not in used:
                    used.add(v)
                    queue.put(v)
        components.append(comp)
    return components

def assign_components_to_splits(
    comps: Sequence[Set[str]],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
    mode: SplitMode = "nodes",
) -> Tuple[List[Set[str]], List[Set[str]], List[Set[str]]]:
    """
    Assign connected components (each a set of node-ids) to train/val/test splits.

    Parameters
    ----------
    comps:
        A sequence of connected components; each component is a set of node IDs.
        Example node IDs: "M:<movie_id>", "N:<name_id>".
    ratios:
        (train, val, test) ratios. Must be positive and sum to ~1.0.
    seed:
        RNG seed used for reproducible shuffling.
    mode:
        - "count": allocate by number of components (simpler).
        - "nodes": allocate greedily by component size (recommended).

    Returns
    -------
    comps_train, comps_val, comps_test:
        Lists of components assigned to each split.

    Notes
    -----
    - "nodes" mode tries to match target ratios by *total number of nodes* in each split.
      This is usually better because components can be very imbalanced in size.
    - For extremely large graphs, you may prefer allocating by edge-count instead;
      that requires edge counts per component.
    """
    if len(comps) == 0:
        return [], [], []

    r_train, r_val, r_test = ratios
    if min(r_train, r_val, r_test) <= 0:
        raise ValueError(f"All ratios must be > 0. Got {ratios}.")
    s = r_train + r_val + r_test
    if not (0.999 <= s <= 1.001):
        # normalize rather than fail hard; keeps function practical
        r_train, r_val, r_test = (r_train / s, r_val / s, r_test / s)

    rng = random.Random(seed)
    comps_list = list(comps)
    rng.shuffle(comps_list)

    if mode == "count":
        n = len(comps_list)
        n_train = int(round(r_train * n))
        n_val = int(round(r_val * n))
        # ensure total doesn't exceed n due to rounding
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        comps_train = comps_list[:n_train]
        comps_val = comps_list[n_train : n_train + n_val]
        comps_test = comps_list[n_train + n_val :]
        return comps_train, comps_val, comps_test

    if mode == "nodes":
        # Greedy bin-packing by component size (descending), to hit target node totals.
        sizes = [len(c) for c in comps_list]
        total_nodes = sum(sizes)
        target_train = r_train * total_nodes
        target_val = r_val * total_nodes
        target_test = r_test * total_nodes

        # Sort by size descending, but keep randomness (shuffle already applied)
        comps_sorted = sorted(comps_list, key=len, reverse=True)

        splits = {
            "train": ([], 0.0, target_train),
            "val": ([], 0.0, target_val),
            "test": ([], 0.0, target_test),
        }
        # Each entry: name -> (list, current_nodes, target_nodes)

        for comp in comps_sorted:
            comp_size = float(len(comp))

            # Choose the split with the largest remaining "need" (target - current),
            # but if all are over target, put into the currently smallest split.
            needs = {
                name: (target - current)
                for name, (_, current, target) in splits.items()
            }
            # Prefer positive needs
            positive = {k: v for k, v in needs.items() if v > 0}

            if positive:
                chosen = max(positive.items(), key=lambda kv: kv[1])[0]
            else:
                chosen = min(splits.items(), key=lambda kv: kv[1][1])[0]  # min current

            lst, current, target = splits[chosen]
            lst.append(comp)
            splits[chosen] = (lst, current + comp_size, target)

        return splits["train"][0], splits["val"][0], splits["test"][0]

    raise ValueError(f"Unknown mode={mode!r}. Use 'count' or 'nodes'.")

def sort_by_entity_type(components_split):
    movies = set()
    names = set()

    for comp in components_split:
        for node in comp:
            if node[:2] == "tt":
                movies.add(node)
            if node[:2] == "nm":
                names.add(node)
    
    return movies,names

movie_to_person = build_relation_map("../data/raw/imdb/title_principals.csv", "tconst", "nconst")
person_to_movie = build_relation_map("../data/raw/imdb/title_principals.csv", "nconst", "tconst")

combined_map = movie_to_person | person_to_movie

for i in range(0,10):
    print(list(movie_to_person.items())[i])
    print(list(person_to_movie.items())[i])



components = find_connected_components(combined_map)

print("COMPONENTS: ")
for i in range(0,2):
    print(components[i])

comp_splits = assign_components_to_splits(components, (0.7,0.1,0.2), 0, "nodes")

print("SPLITS")
print(len(comp_splits[0]))
print(len(comp_splits[1]))
print(len(comp_splits[2]))


# TRAIN ENTITIES BY TYPE
movies_train, names_train = sort_by_entity_type(comp_splits[0])
movies_valid, names_valid = sort_by_entity_type(comp_splits[1])
movies_test, names_test = sort_by_entity_type(comp_splits[2])

print(len(movies_train))
print(len(movies_valid))
print(len(movies_test))
print(len(names_train))
print(len(names_valid))
print(len(names_test))

