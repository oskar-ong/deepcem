import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Tuple

from evaluate import calc_metrics
from experiment_config import DITTO_CONFIG


def finetune(configs_path, task, dataset_dir, log, special_tokens):

    # ================================================================================
    # --- Update ditto/configs.json with datasets ---
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
            "testset": test_fp,
            "tokens": special_tokens
        }

        file_data = [entry for entry in file_data if entry.get("name") != task]

        file_data.append(new_config_entry)

        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)

        log.info(f"Config for '{task}' has been updated/added.")

    # ================================================================================
    # --- Finetune ---
    # ================================================================================
    shutil.copyfile(configs_path, 'configs.json')
    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", task,
        "--batch_size", f"{DITTO_CONFIG['batch_size']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--lr", f"{DITTO_CONFIG['learning_rate']}",
        "--n_epochs", f"{DITTO_CONFIG['epochs']}",
        "--finetuning",
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
        "--run_id", f"{DITTO_CONFIG['seed']}",
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env)
    log.info(f"Finished Finetuning for task: {task}")


def refinetune(configs_path, task, dataset_dir, base, log, special_tokens):

    # ================================================================================
    # --- Update ditto/configs.json with datasets ---
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
            "testset": test_fp,
            "tokens": special_tokens
        }

        file_data = [entry for entry in file_data if entry.get("name") != task]

        file_data.append(new_config_entry)

        with Path(configs_path).open("w", encoding="utf-8") as f:
            json.dump(file_data, f, indent=4)

        log.info(f"Config for '{task}' has been updated/added.")

    # ================================================================================
    # --- Finetune ---
    # ================================================================================
    shutil.copyfile(configs_path, 'configs.json')
    cmd = [
        "python",
        f"./models/ditto/train_ditto.py",
        "--task", task,
        "--batch_size", f"{DITTO_CONFIG['batch_size']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--lr", f"{DITTO_CONFIG['learning_rate']}",
        "--n_epochs", f"{DITTO_CONFIG['epochs']}",
        "--finetuning",
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--checkpoint_path", f"./models/ditto/checkpoints/{base}/model.pt",
        "--fp16",
        "--save_model",
        "--logdir", "./models/ditto/checkpoints/",
        "--run_id", f"{DITTO_CONFIG['seed']}",
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env)
    log.info(f"Finished Finetuning for task: {task}")


def evaluate(task, input_path, output_path, log, true_fp: str) -> Tuple[float, float, float, float]:

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
        "--lm", f"{DITTO_CONFIG['lm']}",
        "--max_len", f"{DITTO_CONFIG['max_len']}",
        "--use_gpu",
        "--fp16",
        "--checkpoint_path", "./models/ditto/checkpoints/",
    ]

    env = os.environ.copy()
    # env["CUDA_VISIBLE_DEVICES"] = "0"
    subprocess.run(cmd, env=env)

    log.info(f"Finished Predicting")

    acc, prec, rec, f1 = calc_metrics(output_path, true_fp)
    acc, prec, rec, f1 = round(acc, 3), round(
        prec, 3), round(rec, 3), round(f1, 3)
    log.info(f"{task} METRICS: Accuracy, Prediction, Recall, F1")
    log.info(f"{task} METRICS: {acc}, {prec}, {rec}, {f1}")
    print(f"{task} {input_path} {output_path} METRICS", acc, prec, rec, f1)
    return acc, prec, rec, f1
