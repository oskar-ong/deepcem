import re

import pandas as pd

from entity_configs.entityConfig import EntityConfig

def create_block_key_movie(row: pd.Series) -> str:
    title = str(row.get("primaryTitle", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_name(row: pd.Series) -> str:
    title = str(row.get("primaryName", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

CONFIGS = [
    # Create config entry for each entity

    # MAIN entity
    EntityConfig(
        name="movie",
        id_col="tconst",
        id_prefix="tt",
        path_basics="./data/raw/imdb/title_basics.csv",
        path_dups="./data/raw/imdb/title_basics_dups.csv",
        path_out_dir="./data/processed/imdb_ref/movie/",
        block_key_func=create_block_key_movie,
        drop_list=['primaryTitle', 'originalTitle', 'cluster_id', 'block_key'],
        is_main=True,
        ditto_dir= "./data/processed/imdb_ref/movie/"
    ),

    # DEPENDENT ENTITY 
    EntityConfig(
        name="name",
        id_col="nconst",
        id_prefix="nm",
        path_basics="./data/raw/imdb/name_basics.csv",
        path_dups="./data/raw/imdb/name_basics_dups.csv",
        path_out_dir="./data/processed/imdb_ref/name/",
        block_key_func=create_block_key_name,
        drop_list=['primaryName', 'cluster_id', 'block_key'],
        rel_csv_path="./data/raw/imdb/title_principals.csv",
        rel_main_col="tconst",
        rel_dep_col="nconst",
        ditto_dir= "./data/processed/imdb_ref/name/"
    ),
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