from dataclasses import dataclass, replace
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
    template_cp: str
    template_conv: str
    relations: List[Relation]
    true_cp_fp: str
    true_test_fp: str

    def resolve_paths(self, dataset: str, pollution: str):
        base_dir = f"./data/{dataset}/{pollution}/inference/{self.name}"

        return replace(
            self,
            template_cp=f"{base_dir}/{self.template_cp}",
            template_conv=f"{base_dir}/{self.template_conv}",
            true_cp_fp=f"{base_dir}/{self.true_cp_fp}",
            true_test_fp=f"{base_dir}/{self.true_test_fp}"
        )


imdb_entities = {

    "movies": EntityConfig(
        name="movies",
        model="imdb_movies_rel_score",
        model_base="imdb_movies",
        id_col="tconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="names", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="nconst", score_col="name_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl"
    ),
    "names": EntityConfig(
        name="names",
        model="imdb_names_rel_score",
        model_base="imdb_names",
        id_col="nconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="movies", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="tconst", score_col="movie_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl"
    )
}

REGISTRY = {
    "imdb_hard": imdb_entities
}
