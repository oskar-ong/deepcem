from ditto_wrapper import evaluate, finetune, refinetune

from logging_setup import setup_logger
from experiment_config import REGISTRY

log = setup_logger("exp_baseline_iterative_imdb_hard")
log.info("Start Experiment: Iterative Matching - IMDB HARD")

def main():

    # ================================================================================
    # GLOBAL CONFIG
    # ================================================================================
    dataset = "imdb_hard"
    config = REGISTRY[dataset]
    configs_path = f"./models/ditto/configs.json"

    # ================================================================================
    # PHASE 1: BASE FINETUNE 
    # ================================================================================

    # ================================================================================
    # MOVIES
    # ================================================================================
    entity = "movie"
    task_movie = config["movies"].model_base
    dataset_dir_movie = f"./data/{dataset}/{entity}/emptyScores"

    finetune(configs_path, task_movie, dataset_dir_movie, log)

    input_path = f"{dataset_dir_movie}/test.txt"
    output_path = f"./ditto_out/{dataset}_{entity}_base_finetune.jsonl"
    evaluate(task_movie, input_path, output_path, log, input_path)

    # ================================================================================
    # NAMES
    # ================================================================================
    
    entity = "name"
    task_name = config["names"].model_base
    dataset_dir_name = f"./data/{dataset}/{entity}/emptyScores"

    finetune(configs_path, task_name, dataset_dir_name, log)

    input_path = f"{dataset_dir_name}/emptyScores/test.txt"
    output_path = f"./ditto_out/{dataset}_{entity}_base_finetune.jsonl"
    evaluate(task_name, input_path, output_path, log, input_path)

    # ================================================================================
    # PHASE 2: RE-TUNE - INCLUDE RELATIONAL SCORES
    # ================================================================================

    # ================================================================================
    # MOVIE REL SCORE
    # ================================================================================
    entity = "movie"
    task_movie_relscore = config["movies"].model
    dataset_dir_movie_injected = f"./data/{dataset}/{entity}/injectedScores"

    refinetune(configs_path, task_movie_relscore, dataset_dir_movie_injected, task_movie, log)

    input_path = f"{dataset_dir_movie_injected}/test.txt"
    output_path = f"./ditto_out/{entity}_injected_finetune.jsonl"
    evaluate(task_movie_relscore, input_path, output_path, log, input_path)

    # ================================================================================
    # NAME REL SCORE
    # ================================================================================
    entity = "name"
    task_name_relscore = config["names"].model
    dataset_dir_name_injected = f"./data/{dataset}/{entity}/injectedScores"

    refinetune(configs_path, task_name_relscore, dataset_dir_name_injected, task_name, log)

    input_path = f"{dataset_dir_name}/test.txt"
    output_path = f"./ditto_out/{entity}_injected_finetune.jsonl"
    evaluate(task_name_relscore, input_path, output_path, log, input_path)

if __name__=="__main__":
    main()