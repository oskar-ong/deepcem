from __future__ import annotations
from collections import defaultdict
import csv
from itertools import combinations
import itertools
from queue import Queue
import random
import re
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

def get_cluster_mapping(uf: UnionFind):
    cluster_mapping = {}

    for entity in uf.parent.keys():
        cluster_mapping[entity] = uf.find(entity)

    return cluster_mapping

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

title_basics = pd.read_csv("../data/raw/imdb/title_basics.csv")
names_basics = pd.read_csv("../data/raw/imdb/name_basics.csv")

movies_subset_train = title_basics[title_basics["tconst"].isin(movies_train)].copy()
movies_subset_valid = title_basics[title_basics["tconst"].isin(movies_valid)].copy()
movies_subset_test = title_basics[title_basics["tconst"].isin(movies_test)].copy()

names_subset_train = names_basics[names_basics["nconst"].isin(names_train)].copy()
names_subset_valid = names_basics[names_basics["nconst"].isin(names_valid)].copy()
names_subset_test = names_basics[names_basics["nconst"].isin(names_test)].copy()

# ========================================================================================
# Subsets are created
# Now create labeled pairs from subsets
# ========================================================================================
movie_dups_uf = build_unionfind_with_singletons(basics_csv="../data/raw/imdb/title_basics.csv", dupes_csv="../data/raw/imdb/title_basics_dups.csv", nconst_col="tconst")
name_dups_uf = build_unionfind_with_singletons(basics_csv="../data/raw/imdb/name_basics.csv", dupes_csv="../data/raw/imdb/name_basics_dups.csv", nconst_col="nconst")

mapping_movies = get_cluster_mapping(movie_dups_uf)
mapping_names = get_cluster_mapping(name_dups_uf)

movies_subset_train['cluster_id'] = movies_subset_train['tconst'].map(mapping_movies)
movies_subset_valid['cluster_id'] = movies_subset_valid['tconst'].map(mapping_movies)
movies_subset_test['cluster_id'] = movies_subset_test['tconst'].map(mapping_movies)

def generate_pairs_for_subset(subset_df, negative_ratio=3):
    """
    Generates Ditto-ready pairs from a subset of movies.
    Target format: (entity1_dict, entity2_dict, label)
    """
    pos_pairs = []
    
    # 1. POSITIVES: The Bottleneck
    # Group by cluster_id to find duplicates within THIS split
    groups = subset_df.groupby('cluster_id')
    for _, group in groups:
        if len(group) > 1:
            # Create all internal combinations (n choose 2)
            for e1, e2 in itertools.combinations(group.to_dict('records'), 2):
                pos_pairs.append((e1, e2, 1))
    
    # 2. NEGATIVES: Filling up
    # We need negative_ratio * len(pos_pairs)
    num_negatives_needed = len(pos_pairs) * negative_ratio
    neg_pairs = generate_hard_negatives(subset_df, num_negatives_needed)
    
    return pos_pairs + neg_pairs

def generate_hard_negatives(df: pd.DataFrame, count):
    neg_pairs = []

    # Sort blocks by size to process smaller, more specific ones first
    block_groups = df.groupby('block_key')
    
    for _, group in block_groups:
        if len(neg_pairs) >= count: 
            break
            
        # Convert to records to access columns easily
        records = group.to_dict('records')
        
        # If the block is too large (e.g., movies starting with "the "), 
        # we sample to avoid N^2 explosion
        if len(records) > 10:
            import random
            random.shuffle(records)
            records = records[:10]

        for e1, e2 in itertools.combinations(records, 2):
            if e1['cluster_id'] != e2['cluster_id']:
                neg_pairs.append((e1, e2, 0))
                print(f"Found Hard Negative: {e1['primaryTitle']} vs {e2['primaryTitle']}")
                if len(neg_pairs) >= count:
                    break
    # If we still need more, fill with random pairs
    while len(neg_pairs) < count:
        s1, s2 = df.sample(2).to_dict('records')
        if s1['cluster_id'] != s2['cluster_id']:
            neg_pairs.append((s1, s2, 0))
            
    return neg_pairs[:count]

def serialize_for_ditto(entity):
    # Exclude the ground truth ID from the model's view
    features = [f"COL {k} VAL {v}" for k, v in entity.items() if k != 'cluster_id']
    return " ".join(features)

def save_to_ditto_file(pairs, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for e1, e2, label in pairs:
            line = f"{serialize_for_ditto(e1)}\t{serialize_for_ditto(e2)}\t{label}\n"
            f.write(line)

def create_block_key_movie(row):

    title = str(row.get("primaryTitle", "")).lower()
    
    # 1. Strip 'the ', 'a ', 'an ' from the start
    title = re.sub(r'^(the|a|an)\s+', '', title)
    
    # 2. Keep only alphanumeric characters
    title = re.sub(r'[^a-z0-9]', '', title)

    return title

movies_train_block = movies_subset_train.copy()
movies_valid_block = movies_subset_valid.copy()
movies_test_block = movies_subset_test.copy()
movies_train_block['block_key'] = movies_train_block.apply(create_block_key_movie, axis=1)
movies_valid_block['block_key'] = movies_valid_block.apply(create_block_key_movie, axis=1)
movies_test_block['block_key'] = movies_test_block.apply(create_block_key_movie, axis=1)

pairs_train_movies= generate_pairs_for_subset(movies_train_block)
pairs_valid_movies= generate_pairs_for_subset(movies_valid_block)
pairs_test_movies= generate_pairs_for_subset(movies_test_block)
#save_to_ditto_file(train_pairs, "train.txt")


from difflib import SequenceMatcher

def get_similarity(a, b):
    return SequenceMatcher(None, str(a), str(b)).ratio()

def analyze_dataset_difficulty(pairs):
    """
    Analyzes the similarity of entities in negative pairs (label 0).
    Pairs format: (entity1_dict, entity2_dict, label)
    """
    neg_similarities = []
    pos_similarities = []

    for e1, e2, label in pairs:
        # We compare the title as it's the most descriptive feature
        sim = get_similarity(e1.get('primaryTitle', ''), e2.get('primaryTitle', ''))
        
        if label == 0:
            neg_similarities.append(sim)
        else:
            pos_similarities.append(sim)

    avg_neg = sum(neg_similarities) / len(neg_similarities) if neg_similarities else 0
    avg_pos = sum(pos_similarities) / len(pos_similarities) if pos_similarities else 0

    print(f"--- Dataset Difficulty Report ---")
    print(f"Average Similarity (Matches):    {avg_pos:.4f}")
    print(f"Average Similarity (Negatives):  {avg_neg:.4f}")
    
    # Interpretation
    if avg_neg > 0.7:
        print("Status: HIGH DIFFICULTY (Good for Fine-tuning)")
    elif avg_neg > 0.4:
        print("Status: MEDIUM DIFFICULTY")
    else:
        print("Status: LOW DIFFICULTY (Too many random negatives)")


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

analyze_dataset_difficulty(pairs_train_movies)
analyze_dataset_difficulty(pairs_valid_movies)
analyze_dataset_difficulty(pairs_test_movies)

write_list_to_csv(pairs_train_movies, "../data/processed/imdb/movie/train.csv")
write_list_to_csv(pairs_valid_movies, "../data/processed/imdb/movie/valid.csv")
write_list_to_csv(pairs_test_movies, "../data/processed/imdb/movie/test.csv")

