from src.ditto_wrapper import evaluate, finetune, refinetune
from src.logging_setup import setup_logger

log = setup_logger("exp_baseline_iterative_imdb_hard")
log.info("Start Experiment: Iterative Matching - IMDB HARD")


def main():

    # ================================================================================
    # GLOBAL CONFIG
    # ================================================================================

    dataset = "imdb_hard"
    configs_path = f"./models/ditto/configs.json"

    # ================================================================================
    # PHASE 1: BASE FINETUNE
    # ================================================================================

    # ================================================================================
    # MOVIES
    # ================================================================================
    entity = "movie"
    task_movie = f"{dataset}_{entity}_iterative"
    dataset_dir_movie = f"./data/{dataset}/{entity}"

    finetune(configs_path, task_movie, dataset_dir_movie, log)

    input_path = f"{dataset_dir_movie}/test.txt"
    output_path = f"./ditto_out/{entity}_iterative.jsonl"
    evaluate(task_movie, input_path, output_path, dataset_dir_movie, log)

    # ================================================================================
    # NAMES
    # ================================================================================

    entity = "name"
    task_name = f"{dataset}_{entity}_iterative"
    dataset_dir_name = f"./data/{dataset}/{entity}"

    finetune(configs_path, task_name, dataset_dir_name, log)

    input_path = f"{dataset_dir_movie}/test.txt"
    output_path = f"./ditto_out/{entity}_iterative.jsonl"
    evaluate(task_movie, input_path, output_path, dataset_dir_movie, log)

    # ================================================================================
    # PHASE 2: RE-TUNE - INCLUDE RELATIONAL SCORES
    # ================================================================================

    # ================================================================================
    # MOVIE REL SCORE
    # ================================================================================
    entity = "movie"
    task_movie_relscore = f"{dataset}_{entity}_iterative_relscore"

    refinetune(configs_path, task_movie_relscore,
               dataset_dir_movie, task_movie, log)

    input_path = f"{dataset_dir_movie}/test_rel_score.txt"
    output_path = f"./ditto_out/{entity}_iterative_rel_score.jsonl"
    evaluate(task_movie_relscore, input_path,
             output_path, dataset_dir_movie, log)

    # ================================================================================
    # NAME REL SCORE
    # ================================================================================
    entity = "name"
    task_name_relscore = f"{dataset}_{entity}_iterative_relscore"

    refinetune(configs_path, task_name_relscore,
               dataset_dir_name, task_name, log)

    input_path = f"{dataset_dir_name}/test_rel_score.txt"
    output_path = f"./ditto_out/{entity}_iterative_rel_score.jsonl"
    evaluate(task_movie_relscore, input_path,
             output_path, dataset_dir_name, log)


if __name__ == "__main__":
    main()
