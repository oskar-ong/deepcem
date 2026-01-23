from csv import DictReader
import csv
from itertools import combinations
import uuid

from deepcem.clustering import UnionFind
from deepcem.serialize import sanitize_value


import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Tuple

def normalize(fp, pk="id", m2m="authors", delimiter=','):
    # MAIN TABLE: PUBLICATIONS
    publications: dict[str, dict] = {}

    # RELATION TABLE: AUTHORS (MENTIONS)
    authors: dict[str, dict] = {}

    # CONNECTION: AUTHOR -> PUBLICATIONS, PUBLICATION -> AUTHORS
    pub_to_authors: dict[str, list] = {}
    author_to_pubs: dict[str, list] = {}

    with open(fp, 'r', encoding='utf-8') as f:
        
        dict_reader = DictReader(f)
        
        list_of_dict = list(dict_reader)

        for d in list_of_dict:
            if d[pk] not in publications:
                pub_key = d[pk]
            else: 
                pub_key = str(uuid.uuid4())

            # assign key-value for each pair in current dict, except primary key and many2many
            publications[pub_key] = {
                k: v for k, v in d.items() if k not in [pk, m2m]
            }    

            # if many2many is assigned as function parameter and m2m value is assigned to current dict
            if m2m in d and d[m2m]:
                # split authors string and assign each stripped name to new value and append to list
                names = [n.strip() for n in d[m2m].split(delimiter) if n.strip()]
                for n in names:
                    a_key = str(uuid.uuid4())
                    authors[a_key] = {"name": n}

                    pub_to_authors.setdefault(pub_key, []).append(a_key)
                    author_to_pubs.setdefault(a_key, []).append(pub_key)
    return publications, authors, pub_to_authors, author_to_pubs

def reduce(pairs, src, out, id_src, id_pairs):
    # 1. Collect all unique IDs from both columns in pairs.csv
    selected_ids = set()
    with open(pairs, 'r', encoding='utf-8') as f:
        reader = DictReader(f)
        for row in reader:
            selected_ids.add(row[id_pairs])

    # 2. Read the full publications file and write only the matches

    with open(src, 'r', encoding='utf-8') as f_in, \
         open(out, 'w', newline='', encoding='utf-8') as f_out:
        
        reader = DictReader(f_in)
        # Automatically use the headers from the source file
        writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            if row[id_src] in selected_ids:
                writer.writerow(row)

def build_ditto_file_from_labels(pairs, table_a, table_b, output_dir, split):

    pairs['ltable_id'] = pairs['ltable_id'].astype(str).str.strip()
    pairs['rtable_id'] = pairs['rtable_id'].astype(str).str.strip()

    pairs['row_left'] = pairs['ltable_id'].map(table_a)
    pairs['row_right'] = pairs['rtable_id'].map(table_b)

    with open(f"{output_dir}/{split}.txt","w", encoding="utf-8") as f:
        
        for index, row in pairs.iterrows():
            data_left = row['row_left']
            data_right = row['row_right']
            label = row['label']
            #print(f"left: {data_left}, right: {data_right}, label: {label}")
            str_left = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in data_left.items())
            str_right = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in data_right.items())
            out_row = f"{str_left}\t{str_right}\t{label}\n"
            f.write(out_row)
    
def enrich_pairs_dataframe(pairs, table_a, table_b) -> list[tuple[dict,dict,int]]:
    """
    Map the attributes to the keys of the split
    """

    df = pairs.copy()

    df['ltable_id'] = df['ltable_id'].astype(str).str.strip()
    df['rtable_id'] = df['rtable_id'].astype(str).str.strip()

    df['row_left'] = df['ltable_id'].map(table_a)
    df['row_right'] = df['rtable_id'].map(table_b)

    candidates = list(zip(df['row_left'], df['row_right'], df['label']))

    enriched_pairs: list[tuple[dict,dict,int]] = []

    for _, row in df.iterrows():
        left_dict = {"id": row['ltable_id'], **row['row_left']}
        right_dict = {"id": row['rtable_id'], **row['row_right']}
        
        enriched_pairs.append((left_dict, right_dict, int(row['label'])))
    
    return enriched_pairs

def save_as_ditto_file(enriched_pairs: list[tuple[dict,dict,int]], output_dir, split):
    """
    Build a ditto file from enriched pairs
    """
    output_path = f"{output_dir}/{split}.txt"
    
    with open(output_path, "w", encoding="utf-8") as f:
        for left, right, label in enriched_pairs:

            # Ditto-specific string serialization logic
            str_left = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in left.items() if k != 'id')
            str_right = " ".join(f"COL {k} VAL {sanitize_value(v)}" for k, v in right.items() if k != 'id')
            
            f.write(f"{str_left}\t{str_right}\t{label}\n")


@dataclass(frozen=True)
class SimpleAuthorFields:
    raw: str
    norm: str
    first_initial: str
    middle_initial: str
    last_name: str
    canon_full: str
    canon_init: str
    block_key: Tuple[str, str]

def _strip_diacritics(s: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", s)
        if not unicodedata.combining(ch)
    )

def normalize_simple_context(name: str) -> SimpleAuthorFields:
    raw = name or ""

    # 1) cleanup: lowercase, strip diacritics, keep only letters/spaces
    s = raw.strip().lower()
    s = _strip_diacritics(s)
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    parts = s.split()
    initials = parts[0] if len(parts) >= 1 else ""
    initials = re.sub(r"[^a-z]", "", initials)

    # 2) robust last name selection: last token (after initials) with len >= 2
    last = ""
    if len(parts) >= 2:
        for tok in reversed(parts[1:]):
            tok = re.sub(r"[^a-z]", "", tok)
            if len(tok) >= 2:
                last = tok
                break

    first_initial = initials[0] if len(initials) >= 1 else ""
    middle_initial = initials[1] if len(initials) >= 2 else ""

    canon_full = f"{last} {initials}".strip()
    canon_init = f"{last} {first_initial}" + (f" {middle_initial}" if middle_initial else "")
    canon_init = canon_init.strip()

    block_key = (last, first_initial)  # if last=="" then block_key is ("", "e") -> easy to ignore later

    return SimpleAuthorFields(
        raw=raw,
        norm=s,
        first_initial=first_initial,
        middle_initial=middle_initial,
        last_name=last,
        canon_full=canon_full,
        canon_init=canon_init,
        block_key=block_key,
    )

def build_block_index(dictionary):
    blocks: Dict[Tuple[str, str], List[str]] = {}

    for mention in dictionary:
        last_name = dictionary[mention]['normalized'].last_name
        if last_name == "":
            continue
        if len(last_name) < 2:
            continue
        
        first_initial = dictionary[mention]['normalized'].first_initial
        blocks.setdefault((last_name, first_initial), []).append(mention)

    return blocks

def match_context_entities(authors, blocks, clusters_authors: UnionFind):

    for k,v in blocks.items():
        for val1, val2 in combinations(v, 2):
            if is_author_match(val1, val2, authors):
                clusters_authors.union(val1, val2)

    return clusters_authors


def is_author_match(a1, a2, authors):
    if authors[a1]['normalized'].canon_full == authors[a2]['normalized'].canon_full:
        return True
    else:
        return False
# Usage
#filter_publications('pairs.csv', 'publications.csv', 'publications_reduced.csv')
