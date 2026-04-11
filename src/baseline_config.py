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
        model="imdb_movie_baseline",
        dir_path=""
    ),
    "name": BaselineConfig(
        name="name",
        model="imdb_name_baseline",
        dir_path=""
    )
}

REGISTRY = {
    "imdb": imdb_entities
}
