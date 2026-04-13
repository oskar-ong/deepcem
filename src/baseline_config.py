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
music_entities = {
    "track": BaselineConfig(
        name="track",
        model="music_track_baseline",
        dir_path=""
    ),
    "artist_credit": BaselineConfig(
        name="artist_credit",
        model="music_artist_credit_baseline",
        dir_path=""
    ),
    "artist": BaselineConfig(
        name="artist",
        model="music_artist_baseline",
        dir_path=""
    ),
    "area": BaselineConfig(
        name="area",
        model="music_area_baseline",
        dir_path=""
    ),
    "label": BaselineConfig(
        name="label",
        model="music_label_baseline",
        dir_path=""
    ),
    "place": BaselineConfig(
        name="place",
        model="music_place_baseline",
        dir_path=""
    ),
    "recording": BaselineConfig(
        name="recording",
        model="music_recording_baseline",
        dir_path=""
    ),
    "medium": BaselineConfig(
        name="medium",
        model="music_medium_baseline",
        dir_path=""
    ),
    "release": BaselineConfig(
        name="release",
        model="music_release_baseline",
        dir_path=""
    ),
    "release_group": BaselineConfig(
        name="release_group",
        model="music_release_group_baseline",
        dir_path=""
    )
}

REGISTRY = {
    "imdb": imdb_entities,
    "music": music_entities
}
