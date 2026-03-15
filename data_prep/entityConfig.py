from dataclasses import dataclass
import re
from typing import Callable, Dict, List

import pandas as pd

@dataclass
class EntityConfig:
    name: str                  # e.g., "movie", "name", "studio"
    id_col: str                # e.g., "tconst", "nconst"
    id_prefix: str             # e.g., "tt", "nm" (used to extract IDs from components)
    path_basics: str
    path_dups: str
    path_out_dir: str          # e.g., "./data/processed/imdb/movie/"
    rels: List[Dict[str, str]]            # e.g. "{"name": "names", "junction_table": "some_path.csv", ""
    block_key_func: Callable
    drop_list: List[str]
    is_main: bool = False
    ditto_dir: str = ""
    
    # Only needed for dependent entities: how they relate to the main entity
    rel_csv_path: str = ""
    rel_main_col: str = ""
    rel_dep_col: str = ""
    rep: str = ""

def create_block_key_movie(row: pd.Series) -> str:
    title = str(row.get("primaryTitle", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_name(row: pd.Series) -> str:
    title = str(row.get("primaryName", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

REGISTRY = {

    "imdb_hard" :
    {
        # MAIN entity
        "movie": EntityConfig(
            name="movie",
            id_col="tconst",
            id_prefix="tt",
            path_basics="./data/raw/imdb_hard/title_basics.csv",
            path_dups="./data/raw/imdb_hard/title_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard/movie/",
            rels = [{"rel_name": "name", "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_movie,
            drop_list=['originalTitle', 'cluster_id', 'block_key'],
            is_main=True,
            ditto_dir= "./data/processed/imdb_hard/movie/"
        ),

        # DEPENDENT ENTITY 
        "name": EntityConfig(
            name="name",
            id_col="nconst",
            id_prefix="nm",
            path_basics="./data/raw/imdb_hard/name_basics.csv",
            path_dups="./data/raw/imdb_hard/name_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard/name/",
            rels = [{"rel_name": "movie", "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_name,
            drop_list=['cluster_id', 'block_key'],
            rel_csv_path="./data/raw/imdb_hard/title_principals.csv",
            rel_main_col="tconst",
            rel_dep_col="nconst",
            ditto_dir= "./data/processed/imdb_hard/name/",
            rep="primaryName"
        )
    }
    # ,

    # "imdb": 
    # [
    #     # MAIN entity
    #     EntityConfig(
    #         name="movie",
    #         id_col="tconst",
    #         id_prefix="tt",
    #         path_basics="./data/raw/imdb/title_basics.csv",
    #         path_dups="./data/raw/imdb/title_basics_dups.csv",
    #         path_out_dir="./data/processed/imdb/movie/",
    #         block_key_func=create_block_key_movie,
    #         drop_list=['primaryTitle', 'originalTitle', 'cluster_id', 'block_key'],
    #         is_main=True,
    #         ditto_dir= "./data/processed/imdb/movie/"
    #     ),

    #     # DEPENDENT ENTITY 
    #     EntityConfig(
    #         name="name",
    #         id_col="nconst",
    #         id_prefix="nm",
    #         path_basics="./data/raw/imdb/name_basics.csv",
    #         path_dups="./data/raw/imdb/name_basics_dups.csv",
    #         path_out_dir="./data/processed/imdb/name/",
    #         block_key_func=create_block_key_name,
    #         drop_list=['primaryName', 'cluster_id', 'block_key'],
    #         rel_csv_path="./data/raw/imdb/title_principals.csv",
    #         rel_main_col="tconst",
    #         rel_dep_col="nconst",
    #         ditto_dir= "./data/processed/imdb/name/"
    #     )
    # ]

}