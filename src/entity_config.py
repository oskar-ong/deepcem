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
    # e.g. "{"name": "names", "junction_table": "some_path.csv", ""
    rels: List[Dict[str, str]]
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


def create_block_key_ability(row: pd.Series) -> str:
    title = str(row.get("ability", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_move(row: pd.Series) -> str:
    title = str(row.get("move", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_item(row: pd.Series) -> str:
    title = str(row.get("item", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_track(row: pd.Series) -> str:
    title = str(row.get("track", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_artist_credit(row: pd.Series) -> str:
    title = str(row.get("artist_credit", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_artist(row: pd.Series) -> str:
    title = str(row.get("artist", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_area(row: pd.Series) -> str:
    title = str(row.get("area", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_label(row: pd.Series) -> str:
    title = str(row.get("label", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_place(row: pd.Series) -> str:
    title = str(row.get("label", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_recording(row: pd.Series) -> str:
    title = str(row.get("recording", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_medium(row: pd.Series) -> str:
    title = str(row.get("medium", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_release(row: pd.Series) -> str:
    title = str(row.get("release", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


def create_block_key_release_group(row: pd.Series) -> str:
    title = str(row.get("release_group", "")).lower()
    title = re.sub(r'^(the|a|an)\s+', '', title)
    return re.sub(r'[^a-z0-9]', '', title)


REGISTRY = {

    "imdb":
    {
        "movie": EntityConfig(
            name="movie",
            id_col="tconst",
            path_basics="./data/raw/imdb/title_basics.csv",
            path_dups="./data/raw/imdb/title_basics_dups.csv",
            path_out_dir="./data/processed/imdb",
            rels=[{"rel_name": "name",
                   "junction_table": "./data/raw/imdb/title_principals.csv"}],
            block_key_func=create_block_key_movie,
            drop_list=['originalTitle'],
        ),
        "name": EntityConfig(
            name="name",
            id_col="nconst",
            path_basics="./data/raw/imdb/name_basics.csv",
            path_dups="./data/raw/imdb/name_basics_dups.csv",
            path_out_dir="./data/processed/imdb",
            rels=[{"rel_name": "movie",
                   "junction_table": "./data/raw/imdb/title_principals.csv"}],
            block_key_func=create_block_key_name,
            drop_list=[]
        )
    },
    "imdb_hard":
    {
        "movie": EntityConfig(
            name="movie",
            id_col="tconst",
            path_basics="./data/raw/imdb_hard/title_basics.csv",
            path_dups="./data/raw/imdb_hard/title_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard",
            rels=[{"rel_name": "name",
                   "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_movie,
            drop_list=['originalTitle'],
        ),
        "name": EntityConfig(
            name="name",
            id_col="nconst",
            path_basics="./data/raw/imdb_hard/name_basics.csv",
            path_dups="./data/raw/imdb_hard/name_basics_dups.csv",
            path_out_dir="./data/processed/imdb_hard",
            rels=[{"rel_name": "movie",
                   "junction_table": "./data/raw/imdb_hard/title_principals.csv"}],
            block_key_func=create_block_key_name,
            drop_list=[]
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
            rels=[
                {"rel_name": "ability",
                    "junction_table": "./data/raw/pokemon/new_ids/poke_ability.csv"},
                {"rel_name": "item",
                    "junction_table": "./data/raw/pokemon/new_ids/poke_item.csv"},
                {"rel_name": "move",
                    "junction_table": "./data/raw/pokemon/new_ids/poke_move.csv"},
                # {"rel_name": "species", "junction_table": "./data/raw/pokemon/50/poke_species.csv"},
            ],
            block_key_func=create_block_key_pokemon,
            drop_list=[],
        ),
        "ability": EntityConfig(
            name="ability",
            id_col="ability",
            path_basics="./data/raw/pokemon/new_ids/ability.csv",
            path_dups="./data/raw/pokemon/new_ids/ability_dups.csv",
            path_out_dir="./data/processed/pokemon/ability/",
            rels=[
                {"rel_name": "pokemon", "junction_table": "./data/raw/pokemon/new_ids/poke_ability.csv"}],
            block_key_func=create_block_key_ability,
            drop_list=[],
        ),
        "item": EntityConfig(
            name="item",
            id_col="item",
            path_basics="./data/raw/pokemon/new_ids/item.csv",
            path_dups="./data/raw/pokemon/new_ids/item_dups.csv",
            path_out_dir="./data/processed/pokemon/item/",
            rels=[
                {"rel_name": "pokemon",
                    "junction_table": "./data/raw/pokemon/new_ids/poke_item.csv"}
            ],
            block_key_func=create_block_key_item,
            drop_list=[],
        ),
        "move": EntityConfig(
            name="move",
            id_col="move",
            path_basics="./data/raw/pokemon/new_ids/move.csv",
            path_dups="./data/raw/pokemon/new_ids/move_dups.csv",
            path_out_dir="./data/processed/pokemon/move/",
            rels=[
                {"rel_name": "pokemon",
                    "junction_table": "./data/raw/pokemon/new_ids/poke_move.csv"}
            ],
            block_key_func=create_block_key_move,
            drop_list=[]
        )
    },
    "music":
        {
        "track": EntityConfig(
            name="track",
            id_col="track",
            path_basics="./data/interim/music/track.csv",
            path_dups="./data/interim/music/track_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "artist_credit",
                    "junction_table": "./data/interim/music/track_artist_credit.csv"},
                {"rel_name": "medium",
                    "junction_table": "./data/interim/music/track_medium.csv"},
                {"rel_name": "recording",
                    "junction_table": "./data/interim/music/track_recording.csv"}
            ],
            block_key_func=create_block_key_track,
            drop_list=[]
        ),
        "artist_credit": EntityConfig(
            name="artist_credit",
            id_col="artist_credit",
            path_basics="./data/interim/music/artist_credit.csv",
            path_dups="./data/interim/music/artist_credit_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "artist",
                    "junction_table": "./data/interim/music/artist_credit_name.csv"},
                {"rel_name": "release",
                    "junction_table": "./data/interim/music/release_artist_credit.csv"},
                {"rel_name": "release_group",
                    "junction_table": "./data/interim/music/release_group_artist_credit.csv"},
                {"rel_name": "recording",
                    "junction_table": "./data/interim/music/recording_artist_credit.csv"}
            ],
            block_key_func=create_block_key_artist_credit,
            drop_list=[]
        ),
        "artist": EntityConfig(
            name="artist",
            id_col="artist",
            path_basics="./data/interim/music/artist.csv",
            path_dups="./data/interim/music/artist_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "artist_credit",
                    "junction_table": "./data/interim/music/artist_credit_name.csv"},
                {"rel_name": "area",
                    "junction_table": "./data/interim/music/artist_area.csv"}
            ],
            block_key_func=create_block_key_artist,
            drop_list=[]
        ),
        "area": EntityConfig(
            name="area",
            id_col="area",
            path_basics="./data/interim/music/area.csv",
            path_dups="./data/interim/music/area_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "artist",
                    "junction_table": "./data/interim/music/artist_area.csv"},
                {"rel_name": "place",
                    "junction_table": "./data/interim/music/place_area.csv"},
                {"rel_name": "label",
                    "junction_table": "./data/interim/music/label_area.csv"}
            ],
            block_key_func=create_block_key_area,
            drop_list=[]
        ),
        "label": EntityConfig(
            name="label",
            id_col="label",
            path_basics="./data/interim/music/label.csv",
            path_dups="./data/interim/music/label_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "area",
                    "junction_table": "./data/interim/music/label_area.csv"}
            ],
            block_key_func=create_block_key_label,
            drop_list=[]
        ),
        "place": EntityConfig(
            name="place",
            id_col="place",
            path_basics="./data/interim/music/place.csv",
            path_dups="./data/interim/music/place_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "area",
                    "junction_table": "./data/interim/music/place_area.csv"}
            ],
            block_key_func=create_block_key_place,
            drop_list=[]
        ),
        "recording": EntityConfig(
            name="recording",
            id_col="recording",
            path_basics="./data/interim/music/recording.csv",
            path_dups="./data/interim/music/recording_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "track",
                    "junction_table": "./data/interim/music/track_recording.csv"},
                {"rel_name": "artist_credit",
                    "junction_table": "./data/interim/music/recording_artist_credit.csv"}
            ],
            block_key_func=create_block_key_recording,
            drop_list=[]
        ),
        "medium": EntityConfig(
            name="medium",
            id_col="medium",
            path_basics="./data/interim/music/medium.csv",
            path_dups="./data/interim/music/medium_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "track",
                    "junction_table": "./data/interim/music/track_medium.csv"},
                {"rel_name": "release",
                    "junction_table": "./data/interim/music/medium_release.csv"}
            ],
            block_key_func=create_block_key_release,
            drop_list=[]
        ),
        "release": EntityConfig(
            name="release",
            id_col="release",
            path_basics="./data/interim/music/release.csv",
            path_dups="./data/interim/music/release_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "medium",
                    "junction_table": "./data/interim/music/medium_release.csv"},
                {"rel_name": "artist_credit",
                    "junction_table": "./data/interim/music/release_artist_credit.csv"},
                {"rel_name": "release_group",
                    "junction_table": "./data/interim/music/release_release_group.csv"},
            ],
            block_key_func=create_block_key_release,
            drop_list=[]
        ),
        "release_group": EntityConfig(
            name="release_group",
            id_col="release_group",
            path_basics="./data/interim/music/release_group.csv",
            path_dups="./data/interim/music/release_group_dups.csv",
            path_out_dir="./data/processed/music",
            rels=[
                {"rel_name": "release",
                    "junction_table": "./data/interim/music/release_release_group.csv"},
                {"rel_name": "artist_credit",
                    "junction_table": "./data/interim/music/release_group_artist_credit.csv"}
            ],
            block_key_func=create_block_key_release_group,
            drop_list=[]
        ),
    }
}
