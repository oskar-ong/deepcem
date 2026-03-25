from dataclasses import dataclass
from typing import List

@dataclass
class Relation:
    name: str
    junction_table: str
    fk: str
    score_col: str  

@dataclass
class EntityConfig:
    name: str                  # e.g., "movie", "name", "studio"
    model: str
    model_base: str
    id_col: str 
    template: str
    relations: List[Relation]  
    true_cp_fp: str
    true_test_fp: str

imdb_entities = {

    "movies": EntityConfig(
        name = "movies",
        model = "imdb_movies_rel_score",
        model_base = "imdb_movies",
        id_col = "tconst",
        template = "./data/imdb_hard/movie/emptyScores/cp_unlabeled.jsonl",
        relations = [Relation(name="names", junction_table="./data/imdb_hard/title_principals.csv", fk="nconst", score_col="name_score")],
        true_cp_fp = f"./data/imdb_hard/movie/emptyScores/cp_labeled.jsonl",
        true_test_fp = f"./data/imdb_hard/movie/emptyScores/test.txt"
        ),
    "names": EntityConfig(
        name = "names",
        model = "imdb_names_rel_score",
        model_base = "imdb_names",
        id_col = "nconst",
        template = "./data/imdb_hard/name/emptyScores/cp_unlabeled.jsonl",
        relations = [Relation(name="movies", junction_table="./data/imdb_hard/title_principals.csv", fk="tconst", score_col="movie_score")],
        true_cp_fp = f"./data/imdb_hard/name/emptyScores/cp_labeled.jsonl",
        true_test_fp = f"./data/imdb_hard/name/emptyScores/test.txt"
        )
    }

REGISTRY = {
    "imdb_hard": imdb_entities
}
