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
            empty_scores_dir=f"./data/{dataset}/{pollution}/emptyScores/{self.name}",
            injected_scores_dir=f"./data/{dataset}/{pollution}/injectedScores/{self.name}"
        )


imdb_entities = {

    "movie": ExperimentConfig(
        name="movie",
        model="imdb_movie_rel_score",
        model_base="imdb_movie",
        id_col="tconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="name", junction_table="./data/imdb/title_principals.csv",
                            fk="nconst", score_col="[NAME_SCORE]")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "name": ExperimentConfig(
        name="name",
        model="imdb_name_rel_score",
        model_base="imdb_name",
        id_col="nconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="movie", junction_table="./data/imdb/title_principals.csv",
                            fk="tconst", score_col="[MOVIE_SCORE]")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    )
}
imdb_hard_entities = {

    "movie": ExperimentConfig(
        name="movie",
        model="imdb_hard_movie_rel_score",
        model_base="imdb_hard_movie",
        id_col="tconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="name", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="nconst", score_col="name_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "name": ExperimentConfig(
        name="name",
        model="imdb_hard_name_rel_score",
        model_base="imdb_hard_name",
        id_col="nconst",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="movie", junction_table="./data/imdb_hard/title_principals.csv",
                            fk="tconst", score_col="movie_score")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    )
}

music_entities = {
    "track": ExperimentConfig(
        name="track",
        model="music_track_rel",
        model_base="music_track",
        id_col="track",
        relations=[Relation(name="artist_credit", junction_table="./data/music/track_artist_credit.csv",
                            fk="artist_credit", score_col="[TRACK_CREDIT_SCORE]"),
                   Relation(name="medium", junction_table="./data/music/track_medium.csv",
                            fk="medium", score_col="[TRACK_MEDIUM_SCORE]"),
                   Relation(name="recording", junction_table="./data/music/track_recording.csv",
                            fk="recording", score_col="[TRACK_RECORDING_SCORE]"),],
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "artist_credit": ExperimentConfig(
        name="artist_credit",
        model="music_artist_credit_rel",
        model_base="music_artist_credit",
        id_col="artist_credit",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="artist", junction_table="./data/music/artist_credit_name.csv",
                     fk="artist", score_col="[ARTIST_CREDIT_ARTIST_SCORE]"),
            Relation(name="release", junction_table="./data/music/release_artist_credit.csv",
                     fk="release", score_col="[ARTIST_CREDIT_RELEASE_SCORE]"),
            Relation(name="release_group", junction_table="./data/music/release_group_artist_credit.csv",
                     fk="release_group", score_col="[ARTIST_CREDIT_RELEASE_GROUP_SCORE]"),
            Relation(name="recording", junction_table="./data/music/recording_artist_credit.csv",
                     fk="recording", score_col="[ARTIST_CREDIT_RECORDING_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "artist": ExperimentConfig(
        name="artist",
        model="music_artist_rel",
        model_base="music_artist",
        id_col="artist",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="artist_credit", junction_table="./data/music/artist_credit_name.csv",
                            fk="artist_credit", score_col="[ARTIST_ARTIST_CREDIT_SCORE]"),
                   Relation(name="area", junction_table="./data/music/artist_area.csv",
                            fk="area", score_col="[ARTIST_AREA_SCORE]")
                   ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "area": ExperimentConfig(
        name="area",
        model="music_area_rel",
        model_base="music_area",
        id_col="area",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="artist", junction_table="./data/music/artist_area.csv",
                     fk="artist", score_col="[AREA_ARTIST_SCORE]"),
            Relation(name="place", junction_table="./data/music/place_area.csv",
                     fk="place", score_col="[AREA_PLACE_SCORE]"),
            Relation(name="label", junction_table="./data/music/label_area.csv",
                     fk="label", score_col="[AREA_LABEL_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "label": ExperimentConfig(
        name="label",
        model="music_label_rel",
        model_base="music_label",
        id_col="label",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="area", junction_table="./data/music/label_area.csv",
                            fk="area", score_col="[LABEL_AREA_SCORE]")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "place": ExperimentConfig(
        name="place",
        model="music_place_rel",
        model_base="music_place",
        id_col="place",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[Relation(name="area", junction_table="./data/music/place_area.csv",
                            fk="area", score_col="[PLACE_AREA_SCORE]")],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "recording": ExperimentConfig(
        name="recording",
        model="music_recording_rel",
        model_base="music_recording",
        id_col="recording",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="track", junction_table="./data/music/track_recording.csv",
                     fk="track", score_col="[RECORDING_TRACK_SCORE]"),
            Relation(name="artist_credit", junction_table="./data/music/recording_artist_credit.csv",
                     fk="artist_credit", score_col="[RECORDING_ARTIST_CREDIT_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "medium": ExperimentConfig(
        name="medium",
        model="music_medium_rel",
        model_base="music_medium",
        id_col="medium",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="track", junction_table="./data/music/track_medium.csv",
                     fk="track", score_col="[MEDIUM_TRACK_SCORE]"),
            Relation(name="release", junction_table="./data/music/medium_release.csv",
                     fk="release", score_col="[MEDIUM_RELEASE_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "release": ExperimentConfig(
        name="release",
        model="music_release_rel",
        model_base="music_release",
        id_col="release",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="medium", junction_table="./data/music/medium_release.csv",
                     fk="medium", score_col="[RELEASE_MEDIUM_SCORE]"),
            Relation(name="artist_credit", junction_table="./data/music/release_artist_credit.csv",
                     fk="artist_credit", score_col="[RELEASE_CREDIT_SCORE]"),
            Relation(name="release_group", junction_table="./data/music/release_release_group.csv",
                     fk="release_group", score_col="[RELEASE_RGROUP_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    ),
    "release_group": ExperimentConfig(
        name="release_group",
        model="music_release_group_rel",
        model_base="music_release_group",
        id_col="release_group",
        template_cp="cp_unlabeled.jsonl",
        template_conv="test_unlabeled.jsonl",
        relations=[
            Relation(name="release", junction_table="./data/music/release_release_group.csv",
                     fk="release", score_col="[RGROUP_RELEASE_SCORE]"),
            Relation(name="artist_credit", junction_table="./data/music/release_group_artist_credit.csv",
                     fk="artist_credit", score_col="[RGROUP_CREDIT_SCORE]")
        ],
        true_cp_fp=f"cp_labeled.jsonl",
        true_test_fp=f"test_labeled.jsonl",
        empty_scores_dir="",
        injected_scores_dir=""
    )

}

REGISTRY = {
    "imdb": imdb_entities,
    "imdb_hard": imdb_hard_entities,
    "music": music_entities
}

DITTO_CONFIG = {
    "batch_size": 32,
    "max_len": 128,
    "learning_rate": 3e-5,
    "epochs": 5,
    "lm": "roberta",
    "seed": 0
}
