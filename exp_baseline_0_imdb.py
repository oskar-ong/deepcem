import json
import os
from pathlib import Path
import shutil
import subprocess

from evaluate import calc_metrics
from logging_setup import setup_logger

log = setup_logger("exp_baseline_0_imdb")
log.info("Start Experiment: Baseline 0 - No Relational Scores")


def finetune(configs_path, task, dataset_dir):

    # ================================================================================
    # Update ditto/configs.json with datasets
    # ================================================================================
    train_fp = f"{dataset_dir}/train.txt"
    valid_fp = f"{dataset_dir}/valid.txt"
    test_fp = f"{dataset_dir}/test.txt"
    configs_path = f"./models/ditto/configs.json"

    with Path(configs_path).open("r", encoding="utf-8") as f:
        file_data: list = json.load(f)

        new_config_entry = {
            "name": task,
            "task_type": "classification",
            "vocab": ["0", "1"],
            "trainset": train_fp,
            "validset": valid_fp,
            "testset": test_fp
        }

        file_data = [entry for entry in file_data if entry.get("name") != task]

        file_data.append(new_config_entry)

        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)
        
        log.info(f"Config for '{task}' has been updated/added.")
        print(f"Config for '{task}' has been updated/added.")

    # ================================================================================
    # Finetune
    # ================================================================================
    shutil.copyfile(configs_path, 'configs.json')
    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", task,
        "--batch_size", "32",
        "--max_len", "128",
        "--lr", "3e-5",
        "--n_epochs", "1",
        "--finetuning",
        "--lm", "roberta",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
    ]

    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env)
    log.info(f"Finished Finetuning for task: {task}")
    print(f"Finished Finetuning for task: {task}")

def evaluate(task, input_path, output_path, dataset_dir):

    log.info("Start Predicting: ")
    log.info(f"Task: {task} ")
    log.info(f"Output Path: {output_path} ")
    log.info(f"Input Path: {input_path} ")

    cmd = [
            "python",
            f"./models/ditto/matcher.py",
            "--task", task,
            "--input_path", input_path,
            "--output_path", output_path,
            "--lm", "roberta",
            "--max_len", "128",
            "--use_gpu",
            "--fp16",
            "--checkpoint_path", "./models/ditto/checkpoints/",
        ]

    env = os.environ.copy()
    #env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env)

    log.info(f"Finished Predicting")

    true_fp = f"{dataset_dir}/test.txt"
    acc, prec, rec, f1 = calc_metrics(output_path, true_fp)
    log.info(f"{task} METRICS: Accuracy, Prediction, Recall, F1")
    log.info(f"{task} METRICS: {round(acc,3)}, {round(prec,3)}, {round(rec,3)}, {round(f1,3)}")
    print(f"{task} METRICS BASELINE 0", acc, prec, rec, f1)

def main():
    dataset = "imdb_hard"
    configs_path = f"./models/ditto/configs.json"

    # ================================================================================
    # MOVIES
    # ================================================================================
    entity = "movie"
    task = f"{dataset}_{entity}_baseline_0"
    dataset_dir = f"./data/{dataset}/{entity}/baseline0"

    finetune(configs_path, task, dataset_dir)

    input_path = f"{dataset_dir}/input.jsonl"
    output_path = f"./ditto_out/{entity}_baseline_0.jsonl"
    evaluate(task, input_path, output_path, dataset_dir)

    # ================================================================================
    # NAMES
    # ================================================================================
    entity = "name"
    task_name = f"{dataset}_{entity}_baseline_0"
    dataset_dir = f"./data/{dataset}/{entity}/baseline0" # TODO: Should build like this: dataset_dir -> task_dir -> entity type

    finetune(configs_path, task_name, dataset_dir)

    input_path = f"{dataset_dir}/input.jsonl"
    output_path = f"./ditto_out/{entity}_baseline_0.jsonl"
    evaluate(task_name, input_path, output_path, dataset_dir)

if __name__=="__main__":
    main()