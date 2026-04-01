from dataclasses import dataclass, replace
from typing import List


@dataclass
class Relation:
    name: str
    junction_table: str
    fk: str
    score_col: str


@dataclass
class ExperimentConfig:
    name: str                  # e.g., "movie", "name", "studio"
    model: str
    model_base: str
    id_col: str
    template_cp: str
    template_conv: str
    relations: List[Relation]
    true_cp_fp: str
    true_test_fp: str
    empty_scores_dir: str
    injected_scores_dir: str

    def resolve_paths(self, dataset: str, pollution: str):
        base_dir = f"./data/{dataset}/{pollution}/inference/{self.name}"

        return replace(
            self,
            model=f"{self.model}_{pollution}",
            model_base=f"{self.model_base}_{pollution}",
            template_cp=f"{base_dir}/{self.template_cp}",
            template_conv=f"{base_dir}/{self.template_conv}",
            true_cp_fp=f"{base_dir}/{self.true_cp_fp}",
            true_test_fp=f"{base_dir}/{self.true_test_fp}",
            empty_scores_dir=f"./data/{dataset}/{pollution}/emtpyScores/{self.name}",
            injected_scores_dir=f"./data/{dataset}/{pollution}/injectedScores/{self.name}"
        )


imdb_entities = {

    "movies": ExperimentConfig(
        name="movies",
        model="imdb_movies_rel_score",
        model_base="imdb_movies",
        id_col="tconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="names", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="nconst", score_col="name_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "names": ExperimentConfig(
        name="names",
        model="imdb_names_rel_score",
        model_base="imdb_names",
        id_col="nconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="movies", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="tconst", score_col="movie_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    )
}

REGISTRY = {
    "imdb_hard": imdb_entities
}

DITTO_CONFIG = {
    "batch_size": 32,
    "max_len": 128,
    "learning_rate": 3e-5,
    "epochs": 5,
    "lm": "roberta"
}
