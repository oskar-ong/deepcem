
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
filtered_df.to_csv("imdb_duplicates.csv", columns=["tconst", "primaryTitle"])


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