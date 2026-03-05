import re

import pandas as pd

from gen_splits import EntityConfig

def create_block_key_pokemon(row: pd.Series) -> str:
    title = str(row.get("pokemon", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_species(row: pd.Series) -> str:
    gen = str(row.get("generation", "")).lower()
    return gen

CONFIGS = [
    # Create config entry for each entity

    # MAIN entity
    EntityConfig(
        name="pokemon",
        id_col="pokemon",
        id_prefix="m",
        path_basics="./data/raw/pokemon/pokemon.csv",
        path_dups="./data/raw/imdb/pokemon_dups.csv",
        path_out_dir="./data/processed/pokemon/pokemon/",
        block_key_func=create_block_key_pokemon,
        drop_list=['cluster_id', 'block_key'],
        is_main=True,
        ditto_dir= "./data/processed/pokemon/pokemon/"
    ),

    # DEPENDENT ENTITY 
    EntityConfig(
        name="species",
        id_col="species",
        id_prefix="d",
        path_basics="./data/raw/imdb/name_basics.csv",
        path_dups="./data/raw/imdb/name_basics_dups.csv",
        path_out_dir="./data/processed/imdb_ref/name/",
        block_key_func=create_block_key_species,
        drop_list=['primaryName', 'cluster_id', 'block_key'],
        rel_csv_path="./data/raw/imdb/title_principals.csv",
        rel_main_col="tconst",
        rel_dep_col="nconst",
        ditto_dir= "./data/processed/imdb_ref/name/"
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