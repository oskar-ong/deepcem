from dataclasses import dataclass
import re
from typing import Callable, Dict, List

import pandas as pd

@dataclass
class EntityConfig:
    name: str                  # e.g., "movie", "name", "studio"
    id_col: str                # e.g., "tconst", "nconst"          
    path_basics: str
    path_dups: str
    path_out_dir: str          # e.g., "./data/processed/imdb/movie/"
    rels: List[Dict[str, str]]            # e.g. "{"name": "names", "junction_table": "some_path.csv", ""
    block_key_func: Callable
    drop_list: List[str]

def create_block_key_movie(row: pd.Series) -> str:
    title = str(row.get("primaryTitle", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_name(row: pd.Series) -> str:
    title = str(row.get("primaryName", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

def create_block_key_pokemon(row: pd.Series) -> str:
    title = str(row.get("pokemon", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)

REGISTRY = {

    "imdb_hard" :
    {
        "movie": EntityConfig(
            name="movie",
            id_col="tconst",
            path_basics="./data/raw/imdb_hard/title_basics.csv",
            path_dups="./data/raw/imdb_hard/title_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard/movie/",
            rels = [{"rel_name": "name", "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_movie,
            drop_list=['originalTitle', 'cluster_id', 'block_key'],
        ),

        # DEPENDENT ENTITY 
        "name": EntityConfig(
            name="name",
            id_col="nconst",
            path_basics="./data/raw/imdb_hard/name_basics.csv",
            path_dups="./data/raw/imdb_hard/name_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard/name/",
            rels = [{"rel_name": "movie", "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_name,
            drop_list=['cluster_id', 'block_key']
        )
    },
    "pokemon":
    {
        "pokemon": EntityConfig(
            name="pokemon",
            id_col="pokemon",
            path_basics="./data/raw/pokemon/new_ids/pokemon.csv",
            path_dups="./data/raw/pokemon/new_ids/pokemon_dups.csv",
            path_out_dir="./data/processed/pokemon/pokemon/",
            rels = [
                {"rel_name": "ability", "junction_table": "./data/raw/pokemon/new_ids/poke_ability.csv"},
                {"rel_name": "item", "junction_table": "./data/raw/pokemon/new_ids/poke_item.csv"},
                {"rel_name": "move", "junction_table": "./data/raw/pokemon/new_ids/poke_move.csv"},
                # {"rel_name": "species", "junction_table": "./data/raw/pokemon/50/poke_species.csv"},
                ],
            block_key_func=create_block_key_pokemon,
            drop_list=['cluster_id', 'block_key'],
        ),
        "ability": EntityConfig(
            name="ability",
            id_col="ability",
            path_basics="./data/raw/pokemon/new_ids/ability.csv",
            path_dups="./data/raw/pokemon/new_ids/ability_dups.csv",
            path_out_dir="./data/processed/pokemon/ability/",
            rels = [
                {"rel_name": "pokemon", "junction_table": "./data/raw/pokemon/new_ids/poke_ability.csv"}],
            block_key_func=create_block_key_pokemon,
            drop_list=['cluster_id', 'block_key'],
        ),
        "item": EntityConfig(
            name="item",
            id_col="item",
            path_basics="./data/raw/pokemon/new_ids/item.csv",
            path_dups="./data/raw/pokemon/new_ids/item_dups.csv",
            path_out_dir="./data/processed/pokemon/item/",
            rels = [
                {"rel_name": "pokemon", "junction_table": "./data/raw/pokemon/new_ids/poke_item.csv"}
                ],
            block_key_func=create_block_key_pokemon,
            drop_list=['cluster_id', 'block_key'],
        ),
        "move": EntityConfig(
            name="move",
            id_col="move",
            path_basics="./data/raw/pokemon/new_ids/move.csv",
            path_dups="./data/raw/pokemon/new_ids/move_dups.csv",
            path_out_dir="./data/processed/pokemon/move/",
            rels = [
                {"rel_name": "pokemon", "junction_table": "./data/raw/pokemon/new_ids/poke_move.csv"}
                ],
            block_key_func=create_block_key_pokemon,
            drop_list=['cluster_id', 'block_key'],
        )
    }
}