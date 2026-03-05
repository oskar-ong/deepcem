
from typing import Dict, List

import pandas as pd

from imdb import build_unionfind_with_singletons

def generate_hard_positives():

    return None

## PHASE 1: PREPERATION & MAPPING
PATH_RAW_TITLE_BASICS     = "../data/raw/imdb/title_basics.csv"
PATH_RAW_TITLE_DUPS       = "../data/raw/imdb/title_basics_dups.csv"
COL_TCONST     = "tconst"
df = pd.read_csv(PATH_RAW_TITLE_BASICS)
uf_movies = build_unionfind_with_singletons(PATH_RAW_TITLE_BASICS, PATH_RAW_TITLE_DUPS, COL_TCONST)
# create cluster id based on unionfind
df['cluster_id'] = df[COL_TCONST].apply(lambda x: uf_movies.find(x))

filtered_df = df[df[COL_TCONST] != df['cluster_id']].copy()
filtered_df.set_index("tconst")
filtered_df.to_csv("imdb_duplicates.csv", columns=["tconst", "primaryTitle"], index=False)
# print(filtered_df.columns)

# Load the datasets
original_df = df.copy()
new_df = pd.read_csv('llmprompt-duplicats.csv')

title_map = new_df.set_index('tconst')['primaryTitle']

# Update "primaryTitle" in original_df where the tconst exists in new_df
duplicate_counts = new_df['tconst'].value_counts()
print(duplicate_counts[duplicate_counts > 1])
original_df['primaryTitle'] = original_df['tconst'].map(title_map).fillna(original_df['primaryTitle'])

# Save the result
original_df.drop(columns=["cluster_id"], inplace=True)
original_df.to_csv('original_updated.csv', index=False)

## PHASE 1: PREPERATION & MAPPING
PATH_RAW_NAME_BASICS     = "../data/raw/imdb/name_basics.csv"
PATH_RAW_NAME_DUPS       = "../data/raw/imdb/name_basics_dups.csv"
COL_NCONST     = "nconst"
df = pd.read_csv(PATH_RAW_NAME_BASICS)
uf_name = build_unionfind_with_singletons(PATH_RAW_NAME_BASICS, PATH_RAW_NAME_DUPS, COL_NCONST)
# create cluster id based on unionfind
df['cluster_id'] = df[COL_NCONST].apply(lambda x: uf_name.find(x))

filtered_df = df[df[COL_NCONST] != df['cluster_id']].copy()
filtered_df.to_csv("imdb_name_duplicates.csv", columns=["nconst", "primaryName"], index=False)
# print(filtered_df.columns)

# Load the datasets
original_df = df.copy()
new_df = pd.read_csv('imdb_name_mutated.csv')

title_map = new_df.set_index('nconst')['primaryName']

# Update "primaryTitle" in original_df where the tconst exists in new_df
duplicate_counts = new_df['nconst'].value_counts()
print(duplicate_counts[duplicate_counts > 1])
original_df['primaryName'] = original_df['nconst'].map(title_map).fillna(original_df['primaryName'])

# Save the result
original_df.drop(columns=["cluster_id"], inplace=True)
original_df.to_csv('imbd_name_basics_mutated.csv', index=False)


# Mutation Rate
MUTATION_RATE = 0.2
# Dictionary where keys are cluster_ids and values are lists of entry names
mapping_m_orig_dups: Dict[str, List[str]] = df.groupby('cluster_id')[COL_TCONST].apply(list).to_dict()



## PHASE 2: PARENT HARDENING

# HARD POSITIVE (HP)
#generate_hard_positives()

# HARD NEGATIVE (HN)
#generate_hard_negatives()

## PHASE 3: CHILD TABLE PROPAGATION

# HP 
# find all children belonging to the orignal parent 
# 