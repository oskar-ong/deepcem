from dataclasses import dataclass
from typing import Callable, List


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