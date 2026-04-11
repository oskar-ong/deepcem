from dataclasses import dataclass, replace
from typing import List


@dataclass
class BaselineConfig:
    name: str                  # e.g., "movie", "name", "studio"
    model: str
    dir_path: str

    def resolve_paths(self, dataset: str, pollution: str):

        return replace(
            self,
            model=f"{self.model}_{pollution}",
            dir_path=f"./data/{dataset}/{pollution}/baseA/{self.name}"
        )


imdb_entities = {

    "movie": BaselineConfig(
        name="movie",
        model="imdb_movie_baseline"

    ),
    "name": BaselineConfig(
        name="name",
        model="imdb_name_baseline",
    )
}

REGISTRY = {
    "imdb": imdb_entities
}

DITTO_CONFIG = {
    "batch_size": 32,
    "max_len": 128,
    "learning_rate": 3e-5,
    "epochs": 5,
    "lm": "roberta"
}
