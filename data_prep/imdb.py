from __future__ import annotations
from collections import defaultdict
import csv
from itertools import combinations
from queue import Queue
import random
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import pandas as pd


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

def build_blocks(
    df: pd.DataFrame,
    id_col: str,
    block_key_fn: Callable[[pd.Series], Iterable[str]]
) -> dict[str, list[str]]:
    """
    Build blocking dictionary.

    Parameters
    ----------
    df : DataFrame
        Entity table (movies or names).
    id_col : str
        Column name containing entity ID.
    block_key_fn : function
        Function that receives a row and returns one or multiple block keys.

    Returns
    -------
    dict: block_key -> list of entity IDs
    """
    blocks = defaultdict(list)

    for _, row in df.iterrows():
        entity_id = row[id_col]
        keys = block_key_fn(row)

        if isinstance(keys, str):
            keys = [keys]

        for key in keys:
            if key is not None:
                blocks[key].append(entity_id)

    return blocks


def candidate_pairs_from_blocks(
    blocks: dict[str, list[str]],
    max_block_size: Optional[int] = None,
    seed: int = 0
) -> Set[Tuple[str, str]]:
    """
    Generate unique unordered candidate pairs from blocks.

    Parameters
    ----------
    blocks : dict
        block_key -> list(entity_ids)
    max_block_size : int or None
        If set, randomly subsample blocks larger than this size.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    set of (id1, id2)
        Unique unordered candidate pairs (id1 < id2)
    """
    rng = random.Random(seed)
    pairs = set()

    for key, ids in blocks.items():

        if len(ids) < 2:
            continue

        ids_local = ids

        # Optional: limit very large blocks
        if max_block_size is not None and len(ids) > max_block_size:
            ids_local = rng.sample(ids, max_block_size)

        for a, b in combinations(ids_local, 2):
            # Ensure canonical ordering to avoid duplicates
            if a < b:
                pairs.add((a, b))
            else:
                pairs.add((b, a))

    return pairs

def movie_block_key(row):
    title = str(row["primaryTitle"]).lower().strip()
    year = row["startYear"]

    if pd.isna(year):
        return None

    prefix = title[:2]  # first chars
    #year_bucket = int(year) // 5  # 5-year buckets

    return f"{prefix}"

def name_block_key(row):
    name = str(row["primaryName"]).lower().strip()

    if pd.isna(name):
        return None

    return name[:2]

class UnionFind:
    def __init__(self) -> None:
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: str) -> str:
        # Iterative path compression
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while x != root:
            p = self.parent[x]
            self.parent[x] = root
            x = p
        return root

    def union(self, a: str, b: str) -> None:
        self.add(a)
        self.add(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def build_unionfind_with_singletons(
    *,
    basics_csv: str,
    dupes_csv: str,
    nconst_col: str = "nconst",
    dupes_delimiter: str = ",",
    dupes_has_header: bool = True,
    comment_prefix: Optional[str] = None,
    strip_whitespace: bool = True,
) -> UnionFind:
    """
    1) Create a singleton set for every nconst in name_basics.csv.
    2) Perform unions based on dupes.csv pairs.
    3) Return the UnionFind object (no cluster materialization).
    """
    uf = UnionFind()

    def clean(s: str) -> str:
        return s.strip() if strip_whitespace else s

    # 1) Add all nconst as singletons
    with open(basics_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or nconst_col not in reader.fieldnames:
            raise ValueError(
                f"Column '{nconst_col}' not found in {basics_csv}. "
                f"Found columns: {reader.fieldnames}"
            )

        for row in reader:
            nconst = row.get(nconst_col, "")
            if nconst is None:
                continue
            nconst = clean(nconst)
            if nconst:
                uf.add(nconst)

    # 2) Union based on duplicate pairs
    with open(dupes_csv, "r", newline="", encoding="utf-8") as f:
        lines = (ln for ln in f if not ln.lstrip().startswith(comment_prefix)) if comment_prefix else f
        reader = csv.reader(lines, delimiter=dupes_delimiter)

        if dupes_has_header:
            next(reader, None)

        for row in reader:
            if not row or len(row) < 2:
                continue
            a, b = clean(row[0]), clean(row[1])
            if not a or not b:
                continue
            uf.union(a, b)

    return uf

def build_duplicate_clusters_from_csv(
    csv_path: str,
    *,
    delimiter: str = ",",
    has_header: bool = True,
    comment_prefix: Optional[str] = None,
    strip_whitespace: bool = True,
) -> Dict[str, Set[str]]:
    """
    Reads a CSV of duplicate pairs with schema: col1,col2 and builds connected-component clusters.

    Notes:
      - Empty / malformed rows are skipped.
      - If has_header=False, the first row will be treated as data.
      - comment_prefix (e.g. "#") skips lines starting with that prefix.
    """
    uf = UnionFind()

    def clean(s: str) -> str:
        return s.strip() if strip_whitespace else s

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        # Optionally skip comment lines manually (csv module doesn't do it)
        lines: Iterable[str]
        if comment_prefix:
            lines = (ln for ln in f if not ln.lstrip().startswith(comment_prefix))
        else:
            lines = f

        reader = csv.reader(lines, delimiter=delimiter)
        if has_header:
            next(reader, None)

        for row in reader:
            if not row or len(row) < 2:
                continue
            a, b = clean(row[0]), clean(row[1])
            if not a or not b:
                continue
            uf.union(a, b)

    # Gather components
    comps: Dict[str, Set[str]] = defaultdict(set)
    for x in uf.parent.keys():
        comps[uf.find(x)].add(x)

    # Name clusters c1, c2, ...
    clusters: Dict[str, Set[str]] = {}
    for i, (_, members) in enumerate(sorted(comps.items(), key=lambda kv: (-len(kv[1]), sorted(kv[1]))), start=1):
        clusters[f"c{i}"] = members

    return clusters, uf

def label_pairs(pairs, entity_to_cluster_id_uf:UnionFind):
    rows = []
    for (a,b) in pairs:
        label = 1 if entity_to_cluster_id_uf.find(a) == entity_to_cluster_id_uf.find(b) else 0
        rows.append((a,b,label))
    return rows

movie_to_person = build_relation_map("../data/raw/imdb/title_principals.csv", "tconst", "nconst")
person_to_movie = build_relation_map("../data/raw/imdb/title_principals.csv", "nconst", "tconst")

combined_map = movie_to_person | person_to_movie

# for i in range(0,10):
#     print(list(movie_to_person.items())[i])
#     print(list(person_to_movie.items())[i])

components = find_connected_components(combined_map)

# print("COMPONENTS: ")
# for i in range(0,2):
#     print(components[i])


# CHECK COMPONENT STATS 

def component_size_stats(comp):
    """
    Returns:
        n_movies, n_names, n_total
    """
    n_movies = sum(1 for x in comp if x[:2]=="tt")
    n_names  = sum(1 for x in comp if x[:2]=="nm")
    n_total  = len(comp)
    return n_movies, n_names, n_total

stats = []

for i, comp in enumerate(components):
    n_movies, n_names, n_total = component_size_stats(comp)
    stats.append((i, n_movies, n_names, n_total))

# Sort by total size descending
stats_sorted = sorted(stats, key=lambda x: x[3], reverse=True)

print("Top 10 largest components:")
for s in stats_sorted[:10]:
    print(f"Comp {s[0]} -> movies={s[1]}, names={s[2]}, total={s[3]}")

total_movies = sum(s[1] for s in stats)
total_names  = sum(s[2] for s in stats)

print("Total movies:", total_movies)
print("Total names :", total_names)

comp_splits = assign_components_to_splits(components, (0.7,0.1,0.2), 0, "nodes")

def split_stats(comps_split):
    movies = sum(sum(1 for x in c if x[:2]=="tt") for c in comps_split)
    names  = sum(sum(1 for x in c if x[:2]=="nm") for c in comps_split)
    total  = sum(len(c) for c in comps_split)
    return movies, names, total


train_stats = split_stats(comp_splits[0])
valid_stats = split_stats(comp_splits[1])
test_stats  = split_stats(comp_splits[2])

print("TRAIN  movies,names,total:", train_stats)
print("VALID  movies,names,total:", valid_stats)
print("TEST   movies,names,total:", test_stats)

# Ratios
total_m = train_stats[0] + valid_stats[0] + test_stats[0]
total_n = train_stats[1] + valid_stats[1] + test_stats[1]

print("Movie ratios:",
      train_stats[0]/total_m,
      valid_stats[0]/total_m,
      test_stats[0]/total_m)

print("Name ratios:",
      train_stats[1]/total_n,
      valid_stats[1]/total_n,
      test_stats[1]/total_n)

largest_comp = stats_sorted[0]
print("Largest component name %:",
      largest_comp[2] / total_names)

# print("SPLITS")
# print(len(comp_splits[0]))
# print(len(comp_splits[1]))
# print(len(comp_splits[2]))

# TRAIN ENTITIES BY TYPE
movies_train, names_train = sort_by_entity_type(comp_splits[0])
movies_valid, names_valid = sort_by_entity_type(comp_splits[1])
movies_test, names_test = sort_by_entity_type(comp_splits[2])

# print(len(movies_train))
# print(len(movies_valid))
# print(len(movies_test))
# print(len(names_train))
# print(len(names_valid))
# print(len(names_test))


# if "tt12542882" in movies_train:
#     if "tt0203189" in movies_train:
#         print("yes")

title_basics = pd.read_csv("../data/raw/imdb/title_basics.csv")
names_basics = pd.read_csv("../data/raw/imdb/name_basics.csv")


movies_subset_train = title_basics[title_basics["tconst"].isin(movies_train)]
movies_subset_valid = title_basics[title_basics["tconst"].isin(movies_valid)]
movies_subset_test = title_basics[title_basics["tconst"].isin(movies_test)]

names_subset_train = names_basics[names_basics["nconst"].isin(names_train)]
names_subset_valid = names_basics[names_basics["nconst"].isin(names_valid)]
names_subset_test = names_basics[names_basics["nconst"].isin(names_test)]



# ========================================================================================
# Subsets are created
# Now create labeled pairs from subsets
# ========================================================================================


movie_dups_uf = build_unionfind_with_singletons(basics_csv="../data/raw/imdb/title_basics.csv", dupes_csv="../data/raw/imdb/title_basics_dups.csv", nconst_col="tconst")
name_dups_uf = build_unionfind_with_singletons(basics_csv="../data/raw/imdb/name_basics.csv", dupes_csv="../data/raw/imdb/name_basics_dups.csv", nconst_col="nconst")


Pair = Tuple[str, str]
LabeledPair = Tuple[str, str, int]

def canon_pair(a: str, b: str) -> Pair:
    return (a, b) if a < b else (b, a)

def build_cluster_to_entities(
    entity_ids: Iterable[str],
    entity_to_cluster: UnionFind,
) -> Dict[str, List[str]]:
    clusters = defaultdict(list)
    for eid in entity_ids:
        cid = entity_to_cluster.find(eid)
        if cid is not None:
            clusters[cid].append(eid)
    return clusters


def generate_positive_pairs(
    cluster_to_entities: Dict[str, List[str]],
    max_pairs_per_cluster: Optional[int] = None,
    seed: int = 0,
) -> Set[Pair]:
    """
    All (or capped) within-cluster pairs.
    """
    rng = random.Random(seed)
    pos: Set[Pair] = set()

    for cid, ids in cluster_to_entities.items():
        if len(ids) < 2:
            continue

        all_pairs = [canon_pair(a, b) for a, b in combinations(ids, 2)]

        if max_pairs_per_cluster is not None and len(all_pairs) > max_pairs_per_cluster:
            all_pairs = rng.sample(all_pairs, max_pairs_per_cluster)

        pos.update(all_pairs)

    return pos








blocks_movie_train = build_blocks(
    df=movies_subset_train,
    id_col="tconst",
    block_key_fn=movie_block_key
)
blocks_movie_valid = build_blocks(
    df=movies_subset_valid,
    id_col="tconst",
    block_key_fn=movie_block_key
)
blocks_movie_test = build_blocks(
    df=movies_subset_test,
    id_col="tconst",
    block_key_fn=movie_block_key
)

blocks_name_train = build_blocks(
    df=names_subset_train,
    id_col="nconst",
    block_key_fn=name_block_key
)
blocks_name_valid = build_blocks(
    df=names_subset_valid,
    id_col="nconst",
    block_key_fn=name_block_key
)
blocks_name_test = build_blocks(
    df=names_subset_test,
    id_col="nconst",
    block_key_fn=name_block_key
)

def generate_labeled_pairs_from_blocks(
    blocks: dict[str, list[str]],
    entity_to_cluster: UnionFind,
    max_block_size: int | None = None,
    seed: int = 0,
):
    rng = random.Random(seed)
    for _, ids in blocks.items():
        if len(ids) < 2:
            continue
        ids_local = ids
        if max_block_size is not None and len(ids) > max_block_size:
            ids_local = rng.sample(ids, max_block_size)

        for a, b in combinations(ids_local, 2):
            if a > b:
                a, b = b, a
            y = 1 if entity_to_cluster.find(a) == entity_to_cluster.find(b) else 0
            yield a, b, y

def sample_pairs_to_budget(
    labeled_pairs_iter,
    max_pos: int,
    max_neg: int,
    seed: int = 0,
):
    """
    Streaming sampler: accept up to max_pos positives and max_neg negatives.
    """
    rng = random.Random(seed)
    pos, neg = [], []

    for a, b, y in labeled_pairs_iter:
        if y == 1:
            if len(pos) < max_pos:
                pos.append((a, b, 1))
        else:
            if len(neg) < max_neg:
                neg.append((a, b, 0))

        if len(pos) >= max_pos and len(neg) >= max_neg:
            break

    # shuffle so order isn't biased by block traversal
    rng.shuffle(pos)
    rng.shuffle(neg)
    return pos + neg



candidate_pairs_movie_train = candidate_pairs_from_blocks(
    blocks_movie_train,
    max_block_size=100,
    seed=42
)

candidate_pairs_movie_valid = candidate_pairs_from_blocks(
    blocks_movie_valid,
    max_block_size=100,
    seed=42
)

candidate_pairs_movie_test = candidate_pairs_from_blocks(
    blocks_movie_test,
    max_block_size=100,
    seed=42
)

candidate_pairs_name_train = candidate_pairs_from_blocks(
    blocks_name_train,
    max_block_size=100,
    seed=42
)

candidate_pairs_name_valid = candidate_pairs_from_blocks(
    blocks_name_valid,
    max_block_size=100,
    seed=42
)

candidate_pairs_name_test = candidate_pairs_from_blocks(
    blocks_name_test,
    max_block_size=100,
    seed=42
)

# print(len(candidate_pairs_movie_train))

# Suppose you want ~100k pairs in val with 1:9 pos:neg
max_pos_val = 10_000
max_neg_val = 90_000

pairs_val = sample_pairs_to_budget(
    generate_labeled_pairs_from_blocks(blocks_movie_valid, movie_dups_uf, max_block_size=100, seed=1),
    max_pos=max_pos_val,
    max_neg=max_neg_val,
    seed=1
)

labeled_movie_train = label_pairs(candidate_pairs_movie_train, movie_dups_uf)
labeled_movie_valid = label_pairs(candidate_pairs_movie_valid, movie_dups_uf)
labeled_movie_test = label_pairs(candidate_pairs_movie_test, movie_dups_uf)

labeled_name_train = label_pairs(candidate_pairs_name_train, name_dups_uf)
labeled_name_valid = label_pairs(candidate_pairs_name_valid, name_dups_uf)
labeled_name_test = label_pairs(candidate_pairs_name_test, name_dups_uf)

def write_list_to_csv(lines, output_path, delimiter=","):
    """
    Writes each element of `lines` as one row in a new CSV file.

    If an element is:
      - a list/tuple → written as a full row
      - a single value → written as a one-column row
    """
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=delimiter)

        writer.writerow(["lid", "rid", "label"])

        for item in lines:
            if isinstance(item, (list, tuple)):
                writer.writerow(item)
            else:
                writer.writerow([item])

write_list_to_csv(labeled_movie_train, "../data/processed/imdb/movie_train.csv")
write_list_to_csv(pairs_val, "../data/processed/imdb/movie_valid.csv")
write_list_to_csv(labeled_movie_test, "../data/processed/imdb/movie_test.csv")

write_list_to_csv(labeled_name_train, "../data/processed/imdb/name_train.csv")
write_list_to_csv(labeled_name_valid, "../data/processed/imdb/name_valid.csv")
write_list_to_csv(labeled_name_test, "../data/processed/imdb/name_test.csv")
