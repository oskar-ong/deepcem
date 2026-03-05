
from collections import defaultdict
from dataclasses import dataclass
import json
import random
from typing import Callable, List

import pandas as pd

from gen_splits import add_labels, assign_components_to_splits, build_relation_map, build_unionfind_with_singletons, create_block_key_movie, create_block_key_name, find_connected_components, generate_pairs_for_subset, process_and_save_ditto, process_relationship_scores, propagate_dependency_pairs, serialize_to_ditto, write_input_json


@dataclass
class EntityConfig:
    name: str                  # e.g., "movie", "name", "studio"
    id_col: str                # e.g., "tconst", "nconst"
    id_prefix: str             # e.g., "tt", "nm" (used to extract IDs from components)
    path_basics: str
    path_dups: str
    path_out_dir: str          # e.g., "./data/processed/imdb/movie/"
    block_key_func: Callable
    drop_list: List[str]
    is_main: bool = False
    ditto_dir: str = ""
    
    # Only needed for dependent entities: how they relate to the main entity
    rel_csv_path: str = ""
    rel_main_col: str = ""
    rel_dep_col: str = ""

ds = "imdb_hardened"

CONFIGS = [
    # Create config entry for each entity

    # MAIN entity
    EntityConfig(
        name="movie",
        id_col="tconst",
        id_prefix="m",
        path_basics=f"./data/raw/{ds}/title_basics.csv",
        path_dups=f"./data/raw/{ds}/title_basics_dups.csv",
        path_out_dir=f"./data/processed/{ds}/movie/",
        block_key_func=create_block_key_movie,
        # drop_list=['primaryTitle', 'originalTitle', 'cluster_id', 'block_key'],
        drop_list=['originalTitle', 'cluster_id', 'block_key'],
        is_main=True,
        ditto_dir= f"./data/processed/{ds}/movie/"
    ),

    # DEPENDENT ENTITY 
    EntityConfig(
        name="name",
        id_col="nconst",
        id_prefix="d",
        path_basics=f"./data/raw/{ds}/name_basics.csv",
        path_dups=f"./data/raw/{ds}/name_basics_dups.csv",
        path_out_dir=f"./data/processed/{ds}/name/",
        block_key_func=create_block_key_name,
        # drop_list=['primaryName', 'cluster_id', 'block_key'],
        drop_list=['cluster_id', 'block_key'],
        rel_csv_path=f"./data/raw/{ds}/title_principals.csv",
        rel_main_col="tconst",
        rel_dep_col="nconst",
        ditto_dir= f"./data/processed/{ds}/name/"
    )
    # ,
    # DEPENDENT ENTITY 2, .., n
    # EntityConfig(
    #     name="name",
    #     id_col="nconst",
    #     id_prefix="nm",
    #     path_basics="./data/raw/imdb/name_basics.csv",
    #     path_dups="./data/raw/imdb/name_basics_dups.csv",
    #     path_out_dir="./data/processed/imdb/name/",
    #     block_key_func=create_block_key_name,
    #     drop_list=['primaryName', 'cluster_id', 'block_key'],
    #     rel_csv_path="./data/raw/imdb/title_principals.csv",
    #     rel_main_col="tconst",
    #     rel_dep_col="nconst"
    # )
]